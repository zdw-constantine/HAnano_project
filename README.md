# Basecalling Pipeline with HAnano model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

This project provides an end-to-end pipeline for nanopore basecalling, featuring a hybrid deep learning architecture combining convolutional networks with recurrent layers for accurate DNA/RNA sequence identification.

---


## 🛠️ Installation
```bash
git clone [repository_url]
cd project_dir
pip install -r requirements.txt
```

---

## 📋 Commands

### 1. Model Training
```bash
python train.py \
--data-dir /path/to/nn_input \
--output-dir /path/to/training_results \
--window-size 2000 \
--num-epochs 5 \
--batch-size 64 \
--checkpoint /path/to/pretrained.pt
```

**Parameters**:
| Argument | Description |
|----------|-------------|
| `--window-size` | Input segment length (2000 recommended) |
| `--checkpoint` | Optional pre-trained model initialization |

---

### 2. Basecalling (Inference)
```bash
python basecall.py \
--fast5-dir /path/to/raw_fast5 \
--checkpoint /path/to/model.pt \
--output-file /path/to/output.fastq \
--batch-size 64
```

**Note**: Output will be in standard FASTQ format with quality scores.

---

### 3. Training Data Preparation
```bash
python data_prepare.py \
--fast5-dir /path/to/source_fast5 \
--output-dir /path/to/processed_data \
--total-files 1 \
--window-size 2000 \
--window-slide 0 \
--n-cores 4 \
--verbose
```

**Processing Options**:
- `--window-slide 0`: Disables overlapping windows
- `--n-cores 4`: Utilizes 4 CPU cores for parallel processing

---
