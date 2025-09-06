# AdMiT: Adaptive Multi-Source Tuning in Dynamic Environments 


## Overview

Large pre-trained models (like Vision Transformers) offer powerful capabilities but are computationally expensive to fine-tune for every new task or data distribution, especially in resource-constrained edge environments. Parameter-Efficient Tuning (PET) methods (e.g., LoRA, Adapters, VPT) mitigate this by tuning only a small fraction of parameters. However, adapting to dynamically shifting, unlabeled data at test time remains challenging.

AdMiT addresses this by:
1.  **Pre-training** a collection of PET modules ![equation](https://latex.codecogs.com/svg.latex?\{\theta_j\}_{j=1}^{N}), each specialized for a different source distribution ![equation](https://latex.codecogs.com/svg.latex?\mathcal{D}_{\mathcal{S}_j}).
2.  During deployment, when faced with a small batch of unlabeled target data ![equation](https://latex.codecogs.com/svg.latex?T%20\sim%20\mathcal{D_T}), **matching** the target distribution to relevant source distributions using Kernel Mean Embeddings (KMEs) to find optimal weights ![equation](https://latex.codecogs.com/svg.latex?w_j).
3.  **Integrating** a sparse subset of the most relevant source PET modules based on these weights to form a tailored module ![equation](https://latex.codecogs.com/svg.latex?\theta(t)) for the current target batch.
4.  Optionally **fine-tuning** the integrated module ![equation](https://latex.codecogs.com/svg.latex?\theta(t)) using techniques like sharpness-aware pseudo-label entropy minimization for enhanced alignment with the target data.

This approach avoids the need for training auxiliary routing networks or extensive hyperparameter tuning during deployment, making it efficient for edge scenarios. The KME-based matching also preserves data privacy as raw source data is not required during adaptation.

## Core Concepts & Formulas

### 1. Kernel Mean Embedding (KME)

KME provides a non-parametric way to represent probability distributions as points in a Reproducing Kernel Hilbert Space (RKHS) ![equation](https://latex.codecogs.com/svg.latex?\mathcal{H}). AdMiT uses KME to measure similarity between the target distribution and source distributions without requiring access to raw source data.

* **KME Definition:** For a distribution ![equation](https://latex.codecogs.com/svg.latex?\mathcal{P}) and a kernel ![equation](https://latex.codecogs.com/svg.latex?k(\cdot,%20\cdot)), the KME ![equation](https://latex.codecogs.com/svg.latex?\mu_k(\mathcal{P})%20\in%20\mathcal{H}) is:
    
    ![equation](https://latex.codecogs.com/svg.latex?\mu_k(\mathcal{P})%20:=%20\mathbb{E}_{x%20\sim%20\mathcal{P}}[k(x,%20\cdot)]%20=%20\int_{x%20\in%20\mathcal{X}}%20k(x,%20\cdot)%20d\mathcal{P}(x))
    
* **Empirical KME:** Given a finite sample ![equation](https://latex.codecogs.com/svg.latex?X%20=%20\{x_n\}_{n=1}^{|X|}%20\sim%20\mathcal{P}), the KME is estimated empirically as:
    
    ![equation](https://latex.codecogs.com/svg.latex?\hat{\mu}(X)%20:=%20\frac{1}{|X|}%20\sum_{n=1}^{|X|}%20k(x_n,%20\cdot))
    
    We use `ApproximateKME` from `kme.py` to potentially approximate ![equation](https://latex.codecogs.com/svg.latex?\hat{\mu}(X)) using a smaller weighted set ![equation](https://latex.codecogs.com/svg.latex?\{%20(z_m,%20\beta_m)%20\}_{m=1}^k) for efficiency and privacy.

### 2. KME-based Module Matching

AdMiT assumes the target distribution ![equation](https://latex.codecogs.com/svg.latex?\mathcal{D_T}) can be approximated by a weighted combination of source distributions ![equation](https://latex.codecogs.com/svg.latex?\mathcal{D_T}%20\approx%20\sum_{j=1}^{N}%20w_j%20\mathcal{D}_{\mathcal{S}_j}). The core idea is to find the weights ![equation](https://latex.codecogs.com/svg.latex?w_j) that minimize the distance between the target KME ![equation](https://latex.codecogs.com/svg.latex?\hat{\mu}(T)) and the weighted sum of source KMEs ![equation](https://latex.codecogs.com/svg.latex?\overline{\mu(S_j)}) (where ![equation](https://latex.codecogs.com/svg.latex?\overline{\mu(S_j)}) are potentially approximations using `ApproximateKME`):

![equation](https://latex.codecogs.com/svg.latex?\min_{\{w_j\}_{j=1}^{N}}%20||%20\hat{\mu}(T)%20-%20\sum_{j=1}^{N}%20w_j%20\overline{\mu(S_j)}%20||_{\mathcal{H}}^2)

This optimization problem (Eqn. 4 in the paper) is solved using quadratic programming (via `cvxopt` in `kme.py`) to find the optimal weights ![equation](https://latex.codecogs.com/svg.latex?\hat{w}_j), which represent the relevance of each source module to the target data.

### 3. Module Integration

Once the top-M weights ![equation](https://latex.codecogs.com/svg.latex?\hat{w}_j) are found, the corresponding source PET module parameters ![equation](https://latex.codecogs.com/svg.latex?\theta_j) (specifically, their state dictionaries) are integrated via weighted averaging to create the target-specific module ![equation](https://latex.codecogs.com/svg.latex?\theta(t)):

![equation](https://latex.codecogs.com/svg.latex?\theta(t)%20=%20\sum_{i=1}^{M}%20\bar{w}_i%20\theta_i)

where ![equation](https://latex.codecogs.com/svg.latex?\bar{w}_i) are the normalized weights of the top-M selected modules. This integration is handled by the static `integrate_modules` method within each PET module class (e.g., `lora.py`, `adapter.py`).

### 4. Sharpness-Aware Tuning (Optional)

To further enhance performance, the integrated module ![equation](https://latex.codecogs.com/svg.latex?\theta(t)) can be fine-tuned on the current target batch ![equation](https://latex.codecogs.com/svg.latex?T^{(t)}%20=%20\{x_i^{(t)}\}_{i=1}^{|T|}) using pseudo-labels ![equation](https://latex.codecogs.com/svg.latex?\hat{y}^{(t)}) derived from the model's own predictions. AdMiT uses sharpness-aware minimization (SAM) applied to the pseudo-label entropy loss ![equation](https://latex.codecogs.com/svg.latex?\mathcal{L}^{(t)}) to find flat minima, improving robustness:

* **Entropy Loss:**
    
    ![equation](https://latex.codecogs.com/svg.latex?\mathcal{L}^{(t)}%20=%20-\mathbb{E}_{\mathcal{D}_{\mathcal{T}}^{(t)}}%20\sum_{c=1}^{K}%20\hat{y}_{c}^{(t)}%20\log(\hat{y}_{c}^{(t)}))
    
* **SAM Objective:**
    
    ![equation](https://latex.codecogs.com/svg.latex?\min_{\lambda}%20\mathcal{L}^{SA(t)}(\lambda)%20\quad%20\text{where}%20\quad%20\mathcal{L}^{SA(t)}(\lambda)%20\triangleq%20\max_{||\epsilon||_2%20\le%20\rho}%20\mathcal{L}^{(t)}(\lambda%20+%20\epsilon))
    
    Here, ![equation](https://latex.codecogs.com/svg.latex?\lambda) represents the trainable parameters of ![equation](https://latex.codecogs.com/svg.latex?\theta(t)). We use the `SAM` optimizer wrapper in `admit_model.py` which approximates the gradient ![equation](https://latex.codecogs.com/svg.latex?\nabla_{\lambda}%20\mathcal{L}^{SA(t)}) via a two-step process.

## Features

* Implementation of the core AdMiT algorithm.
* Support for multiple PET methods (LoRA, Adapter, VPT placeholders provided) via `admit_src/pet_modules/`.
* KME-based module matching using `ApproximateKME` approximation (`admit_src/kme.py`).
* Module integration via weighted averaging of PET parameters.
* Optional test-time tuning using Sharpness-Aware Minimization (SAM) on pseudo-label entropy loss (`admit_src/admit_model.py`).
* Scripts for pre-training source modules (`train_source.py`) and performing test-time adaptation (`adapt_target.py`).

## Prerequisites

* Python 3.x
* PyTorch
* TorchVision
* NumPy
* SciPy
* scikit-learn
* CVXOPT (for KME coefficient estimation with constraints)
* tqdm
* Pillow
* Requests

*(Note: CVXOPT may require system libraries like BLAS/LAPACK and build tools. Refer to requirements.txt for suggested versions)*

## Directory Structure
```
admit/
├── data/                     # Directory for datasets
├── admit_src/                # Source code
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── feature_extractor.py  # Feature extraction
│   ├── kme.py                # KME implementation
│   ├── pet_modules/          # PET module implementations
│   ├── admit_model.py        # Core AdMiT model and adaptation logic
│   ├── train_source.py       # Script for pre-training source modules
│   ├── adapt_target.py       # Script for test-time adaptation
│   ├── utils.py              # Utility functions
│   └── config.py             # Configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file

```
## Usage

### 1. Data Preparation

* Download required datasets (e.g., CIFAR-100-C, Digits-Five). 
* Ensure the KME feature extractor model (e.g., DenseNet201 weights specified in `config.py`) is accessible.

### 2. Pre-training Source Modules & KME Generation

* Configure settings in `admit_src/config.py` (paths, models, hyperparameters).
* Run the source training script for desired source domains (e.g., specific corruptions for CIFAR-100-C):
    ```bash
    # Example using script from scripts/ directory
    bash scripts/run_cifar100c_lora.sh # (Modify script for desired sources/targets)

    # Or run directly:
    python admit_src/train_source.py \
        --dataset <source_dataset_name> \
        --domains <list_of_source_domain_ids> \
        --pet_method <lora|adapter|vpt> \
        --output_dir <path_to_save_outputs> \
        # Add other overrides as needed (e.g., --epochs, --lr)
    ```
* This script saves the trained PET state dicts (e.g., `output/pet_modules/cifar100c_fog_sev1_lora.pt`) and KME representations (e.g., `output/kmes/cifar100c_fog_sev1_kme.pt`).

### 3. Test-Time Adaptation

* Ensure target dataset is available.
* Ensure source modules and KMEs from step 2 are in the directory specified by `--source_dir`.
* Run the target adaptation script:
    ```bash
    # run directly:
    python admit_src/adapt_target.py \
        --target_dataset <target_dataset_name> \
        --target_domains <list_of_target_domain_ids> \
        --pet_method <lora|adapter|vpt> \
        --source_dir <path_containing_saved_outputs> \
        --output_dir <path_to_save_results> \
        [--use_tuning | --no_tuning] # Specify tuning mode (AdMiT vs AdMiT-ZeroShot)
        # Add other overrides as needed
    ```
* This script performs batch-wise adaptation using `AdMiTModel` and logs the performance on the target domain(s).

## Citation

If you use this code or the AdMiT method in your research, please cite the original paper:

```bibtex
@inproceedings{chang2024admit,
  title={AdMiT: Adaptive Multi-Source Tuning in Dynamic Environments},
  author={Chang, Xiangyu and Niloy, Fahim Faisal and Ahmed, Sk Miraj and Krishnamurthy, Srikanth V and Guler, Basak and Swami, Ananthram and Oymak, Samet and Roy-Chowdhury, Amit},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
