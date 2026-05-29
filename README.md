# Spectral Edge Encoding - SEE

This repository provides the official implementation of the paper:

**Spectral Edge Encoding - SEE: Does Structural Information Really Enhance Graph Transformer Performance?**
Seungjun Lee, San Kim, Johyeon Kim, Jaekwang Kim
Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM 2025), Seoul, Republic of Korea.
DOI: 10.1145/3746252.3760906

## Overview

Spectral Edge Encoding (SEE) introduces a parameter-free, structure-aware edge encoding framework for Graph Transformers. SEE quantifies each edge’s contribution to the global graph structure by measuring low-frequency spectral shifts in the graph Laplacian eigenvalues. These edge sensitivity scores are injected into Transformer attention logits as a spectral structural bias.

When applied to the Moiré Graph Transformer (MoiréGT), SEE consistently improves performance on seven MoleculeNet molecular property prediction benchmarks. MoiréGT+SEE achieves an average ROC-AUC of 85.3%, outperforming strong graph-based and chemical language model baselines while preserving molecular topology and enabling edge-level interpretability.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{lee2025see,
  title={Spectral Edge Encoding - SEE: Does Structural Information Really Enhance Graph Transformer Performance?},
  author={Lee, Seungjun and Kim, San and Kim, Johyeon and Kim, Jaekwang},
  booktitle={Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)},
  year={2025},
  publisher={ACM},
  doi={10.1145/3746252.3760906}
}
```

# SEE
spectral edge encoding
**Data Preprocessing**.  

Place the following three scripts in the `data/moleculenet_data/` directory:

* `moleculenet_big_data_2d.py`
  – Alternative preprocessing for environments with limited storage.

---

**Usage**

Run training with default hyperparameters and the BBBP dataset by executing:

```bash
python train.py \
  --DATASET BBBP \
  --LEARNING_RATE 0.0005 \
  --BATCH_SIZE 64 \
  --DEPTH 4 \
  --MLP_DIM 256 \
  --HEADS 12 \
  --T_MAX 200 \
  --WEIGHT_DECAY 1e-5 \
  --SCALE_MIN 0.5 \
  --SCALE_MAX 1.5 \
  --device cuda \
  --VERBOSE True
```

Alternatively, you can edit the `CONFIG` dictionary directly in the code and then run:

```bash
python train.py
```

Adjust any of the command‐line arguments to suit your needs.
