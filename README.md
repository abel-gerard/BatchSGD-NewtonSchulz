# Experiment with SGD and Newton-Schulz

## Installation

Create a virtual environment and install the requirements:
```bash
/usr/bin/python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Usage

The source code is in [experiment.ipynb](experiment.ipynb) with the backend written in Julia inside the [batch_sgd_ns.jl](batch_sgd_ns.jl) file. So you may also use the backend independently in Julia. The backend is inspired by the Muon algorithm found [here](https://github.com/KellerJordan/Muon/blob/master/muon.py).

The objective was to demonstrate the effect of Newton-Schulz normalization on the training of a simple linear model (any dense layer without activation function) using Batch Stochastic Gradient Descent (SGD) with momentum.

I first run Newton-Schulz on some matrix to show the effect of the normalization. Then I create a synthetic dataset and train a linear model on it using SGD, comparing the training loss with and without Newton-Schulz normalization applied to the update. Try changing the hyperparameters or the coefficients used in the Newton-Schulz algorithm to see how it might affect the training!

If the notebook gets stuck trying to load the Julia package, you may need to restart the kernel or your IDE.
