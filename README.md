# Information-Theoretic Precision & Recall Metric for Generative Models

This repository contains code and supplementary materials for the paper *A Unifying Information-theoretic Perspective on Evaluating Generative Models*. It includes implementations for Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), and Recall Entropy (RE) metrics, along with scripts to reproduce the experimental results.

## Repository Structure
- **metrics/**: Implementation of the metrics.
- **experiments/**: Scripts and configurations for running experiments.
- **notebooks/**: Jupyter notebooks demonstrating usage and visualizations.
- **docs/**: Supplementary documentation and the paper PDF.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unified-generative-metrics.git

**Authors:**

- **Alexis Fox** ([alexis.fox@duke.edu](mailto:alexis.fox@duke.edu))  
  *Duke University*
- **Samarth Swarup** ([swarup@virginia.edu](mailto:swarup@virginia.edu))  
  *University of Virginia*
- **Abhijin Adiga** ([abhijin@virginia.edu](mailto:abhijin@virginia.edu))  
  *University of Virginia*

**Conference:** AAAI25

**Abstract:**

> Considering the difficulty of interpreting generative model output, there is significant current research focused on determining meaningful evaluation metrics. Several recent approaches utilize “precision” and “recall,” borrowed from the classification domain, to individually quantify the output fidelity (realism) and output diversity (representation of the real data variation), respectively. With the increase in metric proposals, there is a need for a unifying perspective, allowing for easier comparison and clearer explanation of their benefits and drawbacks. To this end, we unify a class of kth-nearest neighbors (kNN)-based metrics under an information-theoretic lens using approaches from kNN density estimation. Additionally, we propose a tri-dimensional metric composed of Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), and Recall Entropy (RE), which separately measure fidelity and two distinct aspects of diversity, inter- and intra-class. Our domain-agnostic metric, derived from the information-theoretic concepts of entropy and cross-entropy, can be dissected for both sample- and mode-level analysis. Our detailed experimental results demonstrate the sensitivity of our metric components to their respective qualities and reveal undesirable behaviors of other metrics.

**ArXiv Link:** [https://arxiv.org/abs/2412.14340](https://arxiv.org/abs/2412.14340)
