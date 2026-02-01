# GraphCare: A Dual-Graph Sepsis Prediction Framework with LLM‑Based Semantic Standardization and Time‑Adaptive Modeling
[![paper](https://img.shields.io/badge/PDF-Available-red?logo=adobeacrobatreader&logoColor=white)](YOUR_PDF_LINK_HERE)

This is the official implementation of our paper "GraphCare: A Dual-Graph Sepsis Prediction Framework with LLM‑Based Semantic Standardization and Time‑Adaptive Modeling" (Accepted for publication). GraphCare is an innovative LLM-guided dual-graph learning framework that fundamentally advances sepsis mortality prediction for elderly patients through three key innovations: multi-lingual semantic embedding for cross-center antibiotic standardization, dual-graph architecture separating static and dynamic physiological relationships, and adaptive temporal modeling for irregular clinical data.

Extensive validation across MIMIC-IV, Stanford ICU, and Zigong datasets demonstrates superior performance, with AUROC up to "0.8842" and F1-score up to "0.8853" across six mortality prediction tasks. GraphCare's robust cross-dataset generalization and comprehensive temporal analysis establish a new paradigm for personalized predictive medicine in critical care.

#### Installation
Create and activate a new conda environment, then install the required packages.
```bash
conda create -n graphcare python=3.9
conda activate graphcare
pip install -r requirements.txt
```

#### Data Pre-processing
Please follow the instructions in the `data/` directory to download and preprocess the MIMIC-IV, Stanford ICU (SIC), and Zigong datasets. Ensure you have the necessary permissions for MIMIC-IV.

After pre-processing, organize your data with the following structure and create corresponding `.csv` files for train/validation/test splits.
```bash
python preprocess/preprocess_mimic.py --data_path /path/to/raw/mimic --output_path /path/to/processed
```

#### Training
To train the GraphCare model (e.g., for 24-hour in-hospital mortality prediction on MIMIC-IV):
```bash
python train.py --dataset MIMIC --prediction_window 24 --task in_hospital_mortality
```
Key arguments:
- `--dataset`: Choose from `MIMIC`, `SIC`, `Zigong`
- `--prediction_window`: Observation window in hours (24 or 48)
- `--task`: Prediction task (`in_hospital_mortality`, `icu_mortality`, `2day`, `3day`, `30day`, `1year`)
- `--llm_mode`: LLM variant (`EnMed`, `ZhMed`, `Mix`)

#### Inference
```python
import torch
from model.graphcare import GraphCare

# Initialize model
model = GraphCare(llm_mode='Mix', dataset='MIMIC')
model.load_state_dict(torch.load('checkpoints/graphcare_mimic_mix.pt'))
model.eval()

# Example input (batch_size=1, channels=1, num_patients=N, time_series_len=T)
# In practice, you would use preprocessed patient data
input_tensor = torch.randn(1, 1, 50, 24)
prediction = model(input_tensor)
```

#### Repository Structure
```
GraphCare/
├── data/               # Data loading and preprocessing scripts
├── model/              # Core model architecture (graphcare.py)
├── train.py            # Main training script
├── eval.py             # Evaluation and inference
├── utils/              # Utility functions and helpers
├── configs/            # Configuration files for experiments
└── requirements.txt    # Python dependencies
```

#### Citation
If you find this work useful, please cite our paper:
```bibtex
@article{cao2025graphcare,
  title={GraphCare: A Dual-Graph Sepsis Prediction Framework with LLM‑Based Semantic Standardization and Time‑Adaptive Modeling},
  author={Cao, Lei and Wang, Hanyu and Wu, Di and Liu, Xiaoli and Wan, Tao and Qin, Zengchang},
  journal={Computer Methods and Programs in Biomedicine},
  year={2025},
  publisher={Elsevier}
}
```

#### License
This project is licensed under the MIT License - see the LICENSE file for details.

#### Acknowledgments
This work was partially supported by the RISE Funding from VinUni–NTU Center for Generative AI Research, Vinuniversity, and the Beijing Capital Health Development Research Project [Grant no. 2024-2-1031].
