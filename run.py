import argparse
from pathlib import Path
import os
import pprint
import numpy as np
import torch

import set_seed
import preprocessors
import losses
from runner import CustomRunner
from model import FFN

"""
Pytorch model (not quantized) training or evaluation (or both simultaneously). 
Example usage:
    1. For training only
        python run.py --data_dir data --lr 1e-1 --n_epochs 32 --batch_size 64 --train --input_size 2 --checkpoint_dir checkpoints
        
    2. For both training and evaluation
        python run.py --data_dir data --lr 1e-1 --n_epochs 32 --batch_size 64 --train --input_size 2 --checkpoint_dir checkpoints --evaluate --eval_data test.npy
        
    3. For evaluation only
        python run.py --data_dir data --eval_data test.npy --batch_size 64 --evaluate --input_size 2 --checkpoint_dir checkpoints --model_name ffn_state_dict.pth
            
"""

parser = argparse.ArgumentParser(description='Run model training or evaluation')

"""
    data args
"""
parser.add_argument('--data_dir', default='data', help='directory with train, validation and test data')
parser.add_argument('--eval_data', default='test.npy', help='file with data for model evaluation (within data_dir)')
parser.add_argument('--input_size', default=2, type=int, help='model input size (without batch size)')
"""
    model args
"""
parser.add_argument('--checkpoint_dir', default='checkpoints', help='model checkpoints directory')
parser.add_argument('--model_name', default=None, help='pretrained model state dict (within checkpoints dir)')
parser.add_argument('--quantized', action='store_true', help='is model quantized or not')
"""
    training args
"""
parser.add_argument('--train', action='store_true', help='train model, if arg specified')
parser.add_argument('--evaluate', action='store_true', help='evaluate model, if arg specified')
parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--n_epochs', default=1, type=int, help='number of training epochs')

args, _ = parser.parse_known_args()

try:
    os.mkdir(args.checkpoint_dir)
except:
    pass

optimizer = None
criterion = None
p = Path(args.checkpoint_dir)

model = FFN(args.input_size)
if args.model_name:
    model = preprocessors.load_from_state_dict(model, p/args.model_name)

if args.train:
    print(f'Running model in train mode for {args.n_epochs} epochs...')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = losses.MSE()
    loaders = preprocessors.make_dataloaders(args.data_dir, 
                                             data_ext='.npy', 
                                             batch_size=args.batch_size)
    
if args.evaluate and not args.train:
    print('Running model evaluation...')
    loaders = {'test': preprocessors.make_dataloader(data_dir=args.data_dir, 
                                                     data_file=args.eval_data, 
                                                     is_train=False, 
                                                     batch_size=args.batch_size)}


cr = CustomRunner(model=model, device=set_seed.DEVICE, input_key='features', input_target_key='targets', 
                  train=args.train, evaluate=args.evaluate, loaders=loaders, 
                  optimizer=optimizer, criterion=criterion, num_epochs=args.n_epochs)
        
if args.train:
    print('Training completed!')
    torch.save(model.state_dict(), p/'ffn_state_dict.pth')
    print(f'Model state dict was saved into \'{args.checkpoint_dir}\' folder.')
if args.evaluate:
    print('Evaluation completed!')
    pprint.pprint(cr.logs, width=5)
