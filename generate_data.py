import argparse
import numpy as np
import os
from pathlib import Path

"""
Synthetic data for model training and evaluation.
Example usage:
    python generate_data.py --n_samples 10000 --output_dir data --seed 42
"""
parser = argparse.ArgumentParser(description='Generate train, validation and test data')
parser.add_argument('--n_samples', type=int, default=100, help='number of samples')
parser.add_argument('--output_dir', type=str, default='data', help='output directory')
parser.add_argument('--seed', type=int, default=42, help='seed for pseudorandom numbers generator')
                    
args, _ = parser.parse_known_args()

np.random.seed(args.seed)

# y = sin(x0) + cos(x1), X ~ standard normal
X = np.random.randn(args.n_samples, 2)
X[:, 0] = np.sin(X[:, 0])
X[:, 1] = np.cos(X[:, 1])
y = np.sum(X, axis=1)

# add noise to X true
noise_factor = 1e-3
X_sample = X + noise_factor * np.random.randn(*X.shape)
data = np.hstack((X_sample, y[:, np.newaxis]))
train_ratio = 0.75
valid_ratio = 0.8

train, valid, test = np.split(data, [int(train_ratio * args.n_samples), 
                                    int(valid_ratio * args.n_samples)])

try:
    os.mkdir(args.output_dir)
except:
    pass
    
p = Path(args.output_dir)
np.save(p/'train.npy', train)
np.save(p/'valid.npy', valid)
np.save(p/'test.npy', test)



