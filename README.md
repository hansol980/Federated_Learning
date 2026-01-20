# Federated Learning Algorithms (MNIST)

This project implements various Federated Learning algorithms using PyTorch and the MNIST dataset. The goal is to compare the performance of different aggregation and optimization strategies under IID and Non-IID data distributions.

## Methods Implemented
The following scripts implement different federated learning strategies with their specific algorithmic characteristics:

1.  **FedAvg (Federated Averaging)**
    *   **Files**: `FedAvg(IID).py`, `FedAvg(Non_IID).py`
    *   **Core Logic**: The baseline Federated Learning algorithm. Clients train locally using SGD, and the server aggregates the weights by simple averaging.
    *   **Aggregation**: $w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_{t+1}^k$
    *   **Note**: `Non_IID` version simulates data heterogeneity by sorting labels and partitioning shards.

2.  **FedProx (Federated Proximal Optimization)**
    *   **File**: `FedProx.py`
    *   **Core Logic**: Introduces a **proximal term** to the local objective function to handle statistical heterogeneity and limit client drift.
    *   **Local Loss**: $L_{local} = L(w) + \frac{\mu}{2} ||w - w_{global}||^2$
    *   **Key Parameter**: `mu` controls the strength of the regularization (keeping local model close to global model).

3.  **SCAFFOLD (Stochastic Controlled Averaging)**
    *   **File**: `Scaffold.py`
    *   **Core Logic**: Uses **Control Variates** ($c$ and $c_i$) to correct the "client drift" in non-IID settings.
    *   **Correction**: During local training, gradients are adjusted: $g_{corrected} = g - c_i + c$
    *   **Updates**: Both model weights and control variates are updated and communicated between client and server.

4.  **FedDyn (Federated Dynamic Regularization)**
    *   **File**: `FedDyn.py`
    *   **Core Logic**: Aligns the global and local stationary points using a dynamic regularizer.
    *   **Local Objective**: Includes a linear penalty term and an L2 term involving the previous round's gradients (history).
    *   **Feature**: Allows the server to maintain a state variable (`server_h`) that helps in faster convergence.

5.  **FedSAM (Federated Sharpness-Aware Minimization)**
    *   **File**: `FedSam.py`
    *   **Core Logic**: Combines Federated Learning with **SAM**. It minimizes both the loss value and the loss sharpness (flatness) to generalize better on heterogeneous data.
    *   **Optimization**: Performs a two-step update:
        1.  **Ascent Step**: Find a weight perturbation $\epsilon$ that maximizes loss.
        2.  **Descent Step**: Update weights to minimize loss at the perturbed point $w + \epsilon$.

## Requirements
To run these scripts, you need Python and the following libraries:
*   Python 3.8+
*   PyTorch
*   Torchvision
*   NumPy

Install dependencies via pip:
```bash
pip install torch torchvision numpy
```

## Usage
You can run each algorithm simply by executing the corresponding python script.

**1. Run FedAvg (IID)**
```bash
python "FedAvg(IID).py"
```

**2. Run FedAvg (Non-IID)**
```bash
python "FedAvg(Non_IID).py"
```

**3. Run FedProx**
```bash
python FedProx.py
```

**4. Run SCAFFOLD**
```bash
python Scaffold.py
```

**5. Run FedDyn**
```bash
python FedDyn.py
```

**6. Run FedSAM**
```bash
python FedSam.py
```

## Experimental Results
Below are the sample execution logs for each method.

### 1. FedAvg -> IID Situation
```
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedAvg\(IID\).py
Device: cpu
Federated Learning Start: 10 rounds, 5 clients
Round 1/10 - Global Accuracy: 91.29%, | Time: 14.67s
Round 2/10 - Global Accuracy: 94.34%, | Time: 13.97s
Round 3/10 - Global Accuracy: 95.78%, | Time: 13.89s
Round 4/10 - Global Accuracy: 96.72%, | Time: 14.06s
Round 5/10 - Global Accuracy: 97.27%, | Time: 14.66s
Round 6/10 - Global Accuracy: 97.39%, | Time: 15.48s
Round 7/10 - Global Accuracy: 97.69%, | Time: 15.46s
Round 8/10 - Global Accuracy: 97.91%, | Time: 16.25s
Round 9/10 - Global Accuracy: 97.99%, | Time: 18.02s
Round 10/10 - Global Accuracy: 98.07%, | Time: 15.35s
Total Training Time: 151.89s
```

### 2. FedAvg -> Non_IID Situation
```
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedAvg\(Non_IID\).py
Device: cpu
Federated Learning Start: 10 rounds, 50 clients
Round 1/10 - Client Avg Loss: 0.8506 | Global Accuracy: 24.78%, | Time: 4.10s
Round 2/10 - Client Avg Loss: 0.4258 | Global Accuracy: 10.46%, | Time: 3.99s
Round 3/10 - Client Avg Loss: 0.3416 | Global Accuracy: 38.82%, | Time: 3.93s
Round 4/10 - Client Avg Loss: 0.3761 | Global Accuracy: 42.23%, | Time: 4.00s
Round 5/10 - Client Avg Loss: 0.3425 | Global Accuracy: 53.07%, | Time: 3.93s
Round 6/10 - Client Avg Loss: 0.2610 | Global Accuracy: 47.60%, | Time: 3.92s
Round 7/10 - Client Avg Loss: 0.3275 | Global Accuracy: 67.26%, | Time: 4.09s
Round 8/10 - Client Avg Loss: 0.1964 | Global Accuracy: 67.45%, | Time: 3.99s
Round 9/10 - Client Avg Loss: 0.2042 | Global Accuracy: 71.12%, | Time: 3.93s
Round 10/10 - Client Avg Loss: 0.1298 | Global Accuracy: 63.39%, | Time: 4.00s
Total Training Time: 39.88s
```

### 3. FedProx -> Non_IID Situation
```
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedProx.py 
Device: cpu
Federated Learning (FedProx) Start: 10 rounds, 50 clients
Round 1/10 - Client Avg Loss: 0.7909 | Global Accuracy: 22.73%, | Time: 4.48s
Round 2/10 - Client Avg Loss: 0.4744 | Global Accuracy: 19.17%, | Time: 4.28s
Round 3/10 - Client Avg Loss: 0.4229 | Global Accuracy: 48.26%, | Time: 4.24s
Round 4/10 - Client Avg Loss: 0.4080 | Global Accuracy: 63.21%, | Time: 4.30s
Round 5/10 - Client Avg Loss: 0.2404 | Global Accuracy: 40.82%, | Time: 4.29s
Round 6/10 - Client Avg Loss: 0.3218 | Global Accuracy: 55.98%, | Time: 4.29s
Round 7/10 - Client Avg Loss: 0.2790 | Global Accuracy: 61.14%, | Time: 4.27s
Round 8/10 - Client Avg Loss: 0.2793 | Global Accuracy: 58.79%, | Time: 4.30s
Round 9/10 - Client Avg Loss: 0.1528 | Global Accuracy: 64.54%, | Time: 4.24s
Round 10/10 - Client Avg Loss: 0.3221 | Global Accuracy: 73.61%, | Time: 4.33s
Total Training Time: 43.02s
```

### 4. Scaffold
```
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python Scaffold.py 
Device: cpu
Federated Learning Start: 10 rounds, 50 clients
Round 1/10 - Client Avg Loss: 0.6151 | Global Accuracy: 18.82%, | Time: 4.04s
Round 2/10 - Client Avg Loss: 0.4347 | Global Accuracy: 34.57%, | Time: 4.05s
Round 3/10 - Client Avg Loss: 0.4695 | Global Accuracy: 47.30%, | Time: 3.97s
Round 4/10 - Client Avg Loss: 0.4132 | Global Accuracy: 59.10%, | Time: 4.01s
Round 5/10 - Client Avg Loss: 0.4555 | Global Accuracy: 70.27%, | Time: 4.27s
Round 6/10 - Client Avg Loss: 0.3519 | Global Accuracy: 66.10%, | Time: 4.04s
Round 7/10 - Client Avg Loss: 0.2561 | Global Accuracy: 71.48%, | Time: 4.25s
Round 8/10 - Client Avg Loss: 0.3226 | Global Accuracy: 78.77%, | Time: 3.97s
Round 9/10 - Client Avg Loss: 0.2529 | Global Accuracy: 79.71%, | Time: 3.97s
Round 10/10 - Client Avg Loss: 0.2483 | Global Accuracy: 83.03%, | Time: 4.03s
Total Training Time: 40.62s
```

### 5. FedDyn 
```
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedDyn.py 
Device: cpu
FedDyn Start: 10 rounds, 50 clients, alpha=0.1
Round 1/10 - Client Avg Loss: 0.8410 | Global Accuracy: 20.57%, | Time: 4.42s
Round 2/10 - Client Avg Loss: 0.3908 | Global Accuracy: 53.17%, | Time: 4.39s
Round 3/10 - Client Avg Loss: 0.3767 | Global Accuracy: 42.95%, | Time: 4.31s
Round 4/10 - Client Avg Loss: 0.4399 | Global Accuracy: 52.27%, | Time: 4.35s
Round 5/10 - Client Avg Loss: 0.3416 | Global Accuracy: 58.29%, | Time: 4.32s
Round 6/10 - Client Avg Loss: 0.1961 | Global Accuracy: 66.67%, | Time: 4.37s
Round 7/10 - Client Avg Loss: 0.2708 | Global Accuracy: 73.13%, | Time: 4.31s
Round 8/10 - Client Avg Loss: 0.1549 | Global Accuracy: 69.00%, | Time: 4.32s
Round 9/10 - Client Avg Loss: 0.1667 | Global Accuracy: 79.69%, | Time: 4.24s
Round 10/10 - Client Avg Loss: 0.1267 | Global Accuracy: 79.43%, | Time: 4.25s
Total Training Time: 43.27s
```

### 6. FedSam
```
(FL) hansol@hansol-ui-MacBookAir Federated_Learning % python FedSam.py 
Device: cpu
FedSAM Experiment Start: 10 rounds, 50 clients, rho=0.05
Round 1/10 - Client Avg Loss: 0.9859 | Global Accuracy: 19.48% | Time: 6.90s
Round 2/10 - Client Avg Loss: 0.5152 | Global Accuracy: 24.16% | Time: 7.34s
Round 3/10 - Client Avg Loss: 0.4937 | Global Accuracy: 35.05% | Time: 6.98s
Round 4/10 - Client Avg Loss: 0.3747 | Global Accuracy: 57.57% | Time: 6.95s
Round 5/10 - Client Avg Loss: 0.3454 | Global Accuracy: 46.87% | Time: 6.89s
Round 6/10 - Client Avg Loss: 0.2291 | Global Accuracy: 55.12% | Time: 6.84s
Round 7/10 - Client Avg Loss: 0.3215 | Global Accuracy: 69.59% | Time: 6.90s
Round 8/10 - Client Avg Loss: 0.2129 | Global Accuracy: 56.65% | Time: 7.58s
Round 9/10 - Client Avg Loss: 0.1682 | Global Accuracy: 73.19% | Time: 7.11s
Round 10/10 - Client Avg Loss: 0.2159 | Global Accuracy: 72.67% | Time: 6.88s
Total Training Time: 70.36s
```