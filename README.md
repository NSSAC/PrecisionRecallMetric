# Information-Theoretic Precision & Recall Metric for Generative Models

This repository contains code and supplementary materials for the paper *A Unifying Information-theoretic Perspective on Evaluating Generative Models*. It includes implementations for our proposed metric **Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), and Recall Entropy (RE)**, along with scripts to produce experimental results.

## Authors

- **Alexis Fox** ([alexis.fox@duke.edu](mailto:alexis.fox@duke.edu))  
  *Duke University*
- **Samarth Swarup** ([swarup@virginia.edu](mailto:swarup@virginia.edu))  
  *University of Virginia*
- **Abhijin Adiga** ([abhijin@virginia.edu](mailto:abhijin@virginia.edu))  
  *University of Virginia*

**Conference:** AAAI25  
**ArXiv Link:** [https://arxiv.org/abs/2412.14340](https://arxiv.org/abs/2412.14340)

## Abstract

> Considering the difficulty of interpreting generative model output, there is significant current research focused on determining meaningful evaluation metrics. Several recent approaches utilize “precision” and “recall,” borrowed from the classification domain, to individually quantify the output fidelity (realism) and output diversity (representation of the real data variation), respectively. With the increase in metric proposals, there is a need for a unifying perspective, allowing for easier comparison and clearer explanation of their benefits and drawbacks. To this end, we unify a class of kth-nearest neighbors (kNN)-based metrics under an information-theoretic lens using approaches from kNN density estimation. Additionally, we propose a tri-dimensional metric composed of Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), and Recall Entropy (RE), which separately measure fidelity and two distinct aspects of diversity, inter- and intra-class. Our domain-agnostic metric, derived from the information-theoretic concepts of entropy and cross-entropy, can be dissected for both sample- and mode-level analysis. Our detailed experimental results demonstrate the sensitivity of our metric components to their respective qualities and reveal undesirable behaviors of other metrics.

<p align="center">
  <a href="https://youtu.be/GP4U6qbah1M">
    <img src="https://img.youtube.com/vi/GP4U6qbah1M/0.jpg" alt="Watch the video" width="600">
  </a>
</p>

## Repository Structure

- **metrics/**: Implementation of our metric and others discussed.
- **notebooks/**: Jupyter notebook for running representative experiments in paper.
- **docs/**: Paper PDF.

## Installation

To set up the repo and test out the metrics:


1. **Clone the repository**
```bash
git clone https://github.com/unified-generative-metrics.git
cd unified-generative-metrics
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Install the package**
```bash
pip install -e .
```

Example Usage:
```python
import numpy as np
from metrics import compute_metrics

num_real_samples = num_fake_samples = 10000
feature_dim = 1000
nearest_k = 5

real_features = np.random.normal(loc=0.0, scale=1.0, size=[num_real_samples, feature_dim])
fake_features = np.random.normal(loc=0.0, scale=1.0, size=[num_fake_samples, feature_dim])

metrics = compute_metrics(real_features, fake_features, nearest_k)

print(metrics)
```
## Citation
To cite our work, use:

```@inproceedings{fox2025unifying,
  title={A Unifying Information-theoretic Perspective on Evaluating Generative Models},
  author={Fox, Alexis and Swarup, Samarth and Adiga, Abhijin},
  booktitle={Proceedings of AAAI25},
  year={2025},
  url={https://arxiv.org/abs/2412.14340}
}
```
