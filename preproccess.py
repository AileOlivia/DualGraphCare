"""
MIMIC Dataset Extraction Script
================================
Extracts clinical data from MIMIC-III or MIMIC-IV (PhysioNet) and produces
a labeled CSV dataset suitable for ML / clinical research tasks.

Requirements
------------
  pip install pandas numpy psycopg2-binary sqlalchemy tqdm

Access
------
  1. Complete CITI training: https://physionet.org/about/citi-course/
  2. Apply for credentialed access at: https://physionet.org/
  3. Download MIMIC-III or MIMIC-IV from PhysioNet or load into PostgreSQL.

Usage
-----
  python mimic_extract.py --mode csv  --mimic_dir /path/to/mimic/csv
  python mimic_extract.py --mode db   --db_uri postgresql://user:pass@localhost/mimiciii
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  –  edit these to change which features / labels are produced
# ──────────────────────────────────────────────────────────────────────────────

# Target label: 'mortality'  | 'los_gt_3'  | 'readmission_30d'
LABEL = "mortality"

# ICU stay window used for feature extraction (hours after admission)
OBSERVATION_WINDOW_HRS = 24

# Vital-sign item IDs  (MIMIC-III chartevents / MIMIC-IV icu/chartevents)
VITAL_ITEMIDS = {
    # Heart rate
    "heart_rate":      [211, 220045],
    # Systolic BP
    "sbp":             [51, 442, 455, 6701, 220179, 220050],
    # Diastolic BP
    "dbp":             [8368, 8440, 8441, 8555, 220180, 220051],
    # SpO2
    "spo2":            [646, 220277],
    # Respiratory rate
    "resp_rate":       [615, 618, 220210, 224690],
    # Temperature (°C)
    "temperature":     [223761, 678],
    # GCS total
    "gcs":             [198, 226755, 220739],
}

# Lab item IDs (labevents)
LAB_ITEMIDS = {
    "creatinine":      [50912],
    "bun":             [51006],
    "wbc":             [51301],
    "hemoglobin":      [51222],
    "platelet":        [51265],
    "sodium":          [50983],
    "potassium":       [50971],
    "bicarbonate":     [50882],
    "lactate":         [50813],
    "glucose":         [50931],
}

# Aggregation statistics applied to each feature over the observation window
AGG_STATS = ["mean", "min", "max", "std", "last"]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv(mimic_dir: Path, table: str, usecols=None) -> pd.DataFrame:
    """Load a MIMIC CSV table from a local directory."""
    candidates = [
        mimic_dir / f"{table}.csv",
        mimic_dir / f"{table}.csv.gz",
        mimic_dir / "hosp" / f"{table}.csv",
        mimic_dir / "icu"  / f"{table}.csv",
        mimic_dir / "hosp" / f"{table}.csv.gz",
        mimic_dir / "icu"  / f"{table}.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            log.info(f"  Loading {path.name} …")
            return pd.read_csv(path, usecols=usecols, low_memory=False)
    raise FileNotFoundError(
        f"Table '{table}' not found under {mimic_dir}. "
        "Check your MIMIC directory structure."
    )


def _load_db(engine, query: str) -> pd.DataFrame:
    return pd.read_sql(query, engine)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – Build cohort (ICU stays)
# ──────────────────────────────────────────────────────────────────────────────

def build_cohort(loader) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per ICU stay:
      icustay_id, subject_id, hadm_id, admittime, intime, outtime,
      age, gender, los_hours, hospital_expire_flag
    """
    log.info("Building patient cohort …")

    # --- ICU stays ---
    icustays = loader("icustays", usecols=[
        "icustay_id", "subject_id", "hadm_id",
        "intime", "outtime", "los",
    ])
    icustays["intime"]  = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (
        (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
    )

    # --- Admissions (for mortality + admit time) ---
    admissions = loader("admissions", usecols=[
        "hadm_id", "admittime", "dischtime",
        "deathtime", "hospital_expire_flag",
        "discharge_location",
    ])
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

    # --- Patients (for age / gender) ---
    patients = loader("patients", usecols=[
        "subject_id", "gender", "dob",
    ])
    patients["dob"] = pd.to_datetime(patients["dob"], errors="coerce")

    # Merge
    cohort = icustays.merge(admissions, on="hadm_id", how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="inner")

    # Age at admission (MIMIC-III stores very old patients as 300 years)
    cohort["age"] = (
        (cohort["admittime"] - cohort["dob"]).dt.days / 365.25
    ).clip(upper=90)

    # Keep adults only (≥18)
    cohort = cohort[cohort["age"] >= 18].copy()

    # First ICU stay per admission
    cohort = (
        cohort.sort_values("intime")
              .drop_duplicates(subset="hadm_id", keep="first")
    )

    log.info(f"  Cohort size: {len(cohort):,} ICU stays")
    return cohort.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – Extract vitals / labs within the observation window
# ──────────────────────────────────────────────────────────────────────────────

def _extract_events(cohort, events_df, itemid_col, time_col,
                    value_col, feature_map, obs_hrs) -> pd.DataFrame:
    """
    Generic extractor for chartevents or labevents.
    Returns wide DataFrame: icustay_id  x  <feature>_<stat>
    """
    # Flatten item-id → feature name lookup
    id2feat = {iid: feat for feat, ids in feature_map.items() for iid in ids}

    # Filter to relevant item IDs
    keep_ids = [i for ids in feature_map.values() for i in ids]
    ev = events_df[events_df[itemid_col].isin(keep_ids)].copy()
    ev["feature"] = ev[itemid_col].map(id2feat)
    ev[time_col]  = pd.to_datetime(ev[time_col])
    ev[value_col] = pd.to_numeric(ev[value_col], errors="coerce")

    # Merge with cohort to get intime
    ev = ev.merge(
        cohort[["icustay_id", "intime"]],
        on="icustay_id", how="inner"
    )

    # Keep only events within the observation window
    ev["hours_from_admit"] = (
        ev[time_col] - ev["intime"]
    ).dt.total_seconds() / 3600
    ev = ev[(ev["hours_from_admit"] >= 0) & (ev["hours_from_admit"] <= obs_hrs)]

    if ev.empty:
        log.warning("  No events found within the observation window.")
        return pd.DataFrame({"icustay_id": cohort["icustay_id"]})

    # Aggregate
    records = []
    for feat, grp in tqdm(ev.groupby("feature"), desc="  Aggregating features"):
        agg = grp.groupby("icustay_id")[value_col].agg(AGG_STATS)
        agg.columns = [f"{feat}_{s}" for s in AGG_STATS]
        records.append(agg)

    wide = pd.concat(records, axis=1).reset_index()
    return wide


def extract_vitals(cohort, loader, obs_hrs) -> pd.DataFrame:
    log.info("Extracting vital signs …")
    try:
        ce = loader("chartevents", usecols=[
            "icustay_id", "itemid", "charttime", "valuenum",
        ])
        return _extract_events(cohort, ce, "itemid", "charttime",
                               "valuenum", VITAL_ITEMIDS, obs_hrs)
    except FileNotFoundError:
        log.warning("  chartevents not found; skipping vitals.")
        return pd.DataFrame({"icustay_id": cohort["icustay_id"]})


def extract_labs(cohort, loader, obs_hrs) -> pd.DataFrame:
    log.info("Extracting lab values …")
    try:
        le = loader("labevents", usecols=[
            "hadm_id", "itemid", "charttime", "valuenum",
        ])
        # labevents uses hadm_id; bridge via cohort
        le = le.merge(cohort[["icustay_id", "hadm_id"]], on="hadm_id", how="inner")
        return _extract_events(cohort, le, "itemid", "charttime",
                               "valuenum", LAB_ITEMIDS, obs_hrs)
    except FileNotFoundError:
        log.warning("  labevents not found; skipping labs.")
        return pd.DataFrame({"icustay_id": cohort["icustay_id"]})


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – Derive labels
# ──────────────────────────────────────────────────────────────────────────────

def derive_labels(cohort: pd.DataFrame) -> pd.DataFrame:
    """Add binary label column based on LABEL config."""
    log.info(f"Deriving label: {LABEL} …")

    if LABEL == "mortality":
        cohort["label"] = cohort["hospital_expire_flag"].astype(int)

    elif LABEL == "los_gt_3":
        cohort["label"] = (cohort["los_hours"] > 72).astype(int)

    elif LABEL == "readmission_30d":
        # Approximate: flag if patient has another admission within 30 days
        # (requires multi-admission check; simplified version here)
        cohort = cohort.sort_values(["subject_id", "admittime"])
        cohort["next_admit"] = cohort.groupby("subject_id")["admittime"].shift(-1)
        cohort["label"] = (
            (cohort["next_admit"] - cohort["dischtime"]).dt.days.between(1, 30)
        ).astype(int)
        cohort.drop(columns="next_admit", inplace=True)

    else:
        raise ValueError(f"Unknown LABEL '{LABEL}'. Choose: mortality, los_gt_3, readmission_30d")

    pos = cohort["label"].sum()
    log.info(f"  Label distribution: {pos:,} positive / {len(cohort)-pos:,} negative "
             f"({100*pos/len(cohort):.1f}% positive)")
    return cohort


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4 – Assemble & clean final dataset
# ──────────────────────────────────────────────────────────────────────────────

STATIC_COLS = ["icustay_id", "subject_id", "hadm_id", "age", "gender", "los_hours", "label"]

def assemble_dataset(cohort, vitals, labs) -> pd.DataFrame:
    log.info("Assembling final dataset …")

    # Encode gender
    cohort["gender"] = (cohort["gender"] == "M").astype(int)  # 1=Male, 0=Female

    base = cohort[STATIC_COLS].copy()
    df = base.merge(vitals, on="icustay_id", how="left")
    df = df.merge(labs,    on="icustay_id", how="left")

    # Drop columns that are >80% missing
    missing_frac = df.isnull().mean()
    drop_cols = missing_frac[missing_frac > 0.80].index.tolist()
    drop_cols = [c for c in drop_cols if c not in STATIC_COLS]
    if drop_cols:
        log.info(f"  Dropping {len(drop_cols)} cols with >80% missing: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    log.info(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIMIC → labeled CSV extractor")
    parser.add_argument("--mode", choices=["csv", "db"], default="csv",
                        help="Data source: local CSV files or PostgreSQL database")
    parser.add_argument("--mimic_dir", type=str, default="./mimic",
                        help="Root directory of MIMIC CSV files (for --mode csv)")
    parser.add_argument("--db_uri", type=str,
                        default="postgresql://mimicuser:mimicpass@localhost/mimiciii",
                        help="SQLAlchemy URI for PostgreSQL (for --mode db)")
    parser.add_argument("--schema", type=str, default="mimiciii",
                        help="DB schema name (e.g. mimiciii or mimiciv)")
    parser.add_argument("--obs_hrs", type=int, default=OBSERVATION_WINDOW_HRS,
                        help=f"Observation window in hours (default: {OBSERVATION_WINDOW_HRS})")
    parser.add_argument("--label", type=str, default=LABEL,
                        choices=["mortality", "los_gt_3", "readmission_30d"],
                        help=f"Target label to generate (default: {LABEL})")
    parser.add_argument("--output", type=str, default="mimic_labeled_dataset.csv",
                        help="Output CSV file path")
    args = parser.parse_args()

    # Override globals from args
    global LABEL, OBSERVATION_WINDOW_HRS
    LABEL = args.label
    OBSERVATION_WINDOW_HRS = args.obs_hrs

    # ── Set up loader ──────────────────────────────────────────────────────────
    if args.mode == "csv":
        mimic_dir = Path(args.mimic_dir)
        if not mimic_dir.exists():
            raise FileNotFoundError(f"MIMIC directory not found: {mimic_dir}")
        loader = lambda table, **kw: _load_csv(mimic_dir, table, **kw)
        log.info(f"Mode: CSV  |  MIMIC dir: {mimic_dir}")

    else:
        from sqlalchemy import create_engine
        engine = create_engine(args.db_uri)
        schema = args.schema
        def loader(table, usecols=None):
            cols = ", ".join(usecols) if usecols else "*"
            return _load_db(engine, f"SELECT {cols} FROM {schema}.{table}")
        log.info(f"Mode: DB   |  URI: {args.db_uri}  schema: {schema}")

    log.info(f"Label: {LABEL}  |  Observation window: {args.obs_hrs}h")

    # ── Pipeline ───────────────────────────────────────────────────────────────
    cohort  = build_cohort(loader)
    cohort  = derive_labels(cohort)
    vitals  = extract_vitals(cohort, loader, args.obs_hrs)
    labs    = extract_labs(cohort, loader, args.obs_hrs)
    dataset = assemble_dataset(cohort, vitals, labs)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    dataset.to_csv(out_path, index=False)
    log.info(f"✓ Dataset saved to: {out_path}")

    # Quick summary
    print("\n── Column overview ──")
    print(dataset.dtypes.to_string())
    print(f"\n── Missing values (top 10) ──")
    print(dataset.isnull().mean().sort_values(ascending=False).head(10).to_string())
    print(f"\n── Label distribution ──")
    print(dataset["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
