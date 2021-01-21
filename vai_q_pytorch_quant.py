import os
from pathlib import Path
import ntpath
import argparse
import warnings

import torch
from pytorch_nndct.apis import torch_quantizer

from tqdm import tqdm
import pprint
from copy import deepcopy

import set_seed
import preprocessors
from model import FFN
from runner import CustomRunner
from losses import MSE


"""
Quantization with vai_q_pytorch

Notes: --fast finetune isn't working now (it's a bug, see my issue https://github.com/Xilinx/Vitis-AI/issues/239)

Example usage:
    1. For float model evaluation
        python vai_q_pytorch_quant.py --data_dir data --calib_data test.npy --input_size 2 --checkpoint_dir checkpoints --model_name ffn_state_dict.pth --batch_size 64 --quant_mode float --evaluate
        
    2. For quantization and evaluation (disable --evaluate if you need quantization only)
        python vai_q_pytorch_quant.py --data_dir data --calib_data test.npy --input_size 2 --checkpoint_dir checkpoints --model_name ffn_state_dict.pth --batch_size 64 --quant_mode calib --evaluate
        
    3. For quantized model evaluation and deployment (set --deploy, if you need .xmodel)
        python vai_q_pytorch_quant.py --data_dir data --calib_data test.npy --input_size 2 --checkpoint_dir checkpoints --model_name ffn_state_dict.pth --batch_size 1 --subset_len 1 --quant_mode test --evaluate --deploy

"""

parser = argparse.ArgumentParser()
"""
    data args
"""
parser.add_argument('--data_dir', default='data', type=str, 
                    help='directory with train, validation and test data')
parser.add_argument('--calib_data', default='test.npy', type=str, 
                    help='file with data for model calibration or evaluation (within data_dir)')
parser.add_argument('--input_size', default=2, type=int, 
                    help='model input size (without batch size)')
"""
    model args
"""
parser.add_argument('--checkpoint_dir', default='checkpoints', type=str, 
                    help='model checkpoints directory')
parser.add_argument('--model_name', default='model.pth', type=str, 
                    help='pretrained model or quantized model state dict (within checkpoint dir)')
"""
    quantization args
"""
parser.add_argument('--subset_len', default=None, type=int, 
                    help='subset_len to evaluate model, using the whole evaluation dataset if it is not set')
parser.add_argument('--batch_size', default=1, type=int, 
                    help='input data batch size to evaluate model')
parser.add_argument('--quant_mode',  default='calib', choices=['float', 'calib', 'test'], 
                    help='quantization mode: float-turn off quantization; '\
                    'calib-calibration of quantization; '\
                    'test-quantized model evaluation')
parser.add_argument('--deploy', dest='deploy', action='store_true', 
                    help='export xmodel for deployment; should be run in \'test\' mode with batch_size=1 and subset_len=1')
parser.add_argument('--evaluate', action='store_true', 
                    help='evaluate model, if arg specified')
parser.add_argument('--fast_finetune', action='store_true',
                    help='fast finetuning on subset_len')

args, _ = parser.parse_known_args()


def eval_loss(model, valid_loader, loss_fn):
    model.eval()
    model = model.to(set_seed.DEVICE)
    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        features = batch['features'].to(set_seed.DEVICE)
        targets = batch['targets'].to(set_seed.DEVICE)
        
        outputs = model(features)
        loss = loss_fn(outputs, targets)

        epoch_loss += loss.item()

    return epoch_loss / len(valid_loader)


def quantization(): 
    if args.quant_mode != 'test' and args.deploy:
        args.deploy = False
        warnings.warn('Exporting xmodel needs to be done in quantization test mode, turn off it in this running!', UserWarning)
        
    if args.quant_mode=='test' and (args.batch_size != 1 or args.subset_len != 1):
        warnings.warn('Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, they\'ll be changed automatically!', UserWarning)
        args.batch_size = 1
        args.subset_len = 1
    
    p = Path(args.checkpoint_dir)/args.model_name
    model = FFN(args.input_size)
    model = preprocessors.load_from_state_dict(model, p)
    

    if args.quant_mode == 'float':
        quant_model = deepcopy(model)
    else:
        rand_input = torch.randn([args.batch_size, args.input_size])
        quantizer = torch_quantizer(args.quant_mode, 
                                    module=deepcopy(model), 
                                    input_args=rand_input, 
                                    bitwidth=8, 
                                    mix_bit=False, 
                                    qat_proc=False, 
                                    device=set_seed.DEVICE)

        quant_model = quantizer.quant_model

    if args.fast_finetune:
        ft_loader = preprocessors.make_dataloader(data_dir=args.data_dir, 
                                                  data_file=args.calib_data,
                                                  subset_len=args.subset_len)
        if args.quant_mode == 'calib':
            loss_fn = MSE().to(set_seed.DEVICE)
            quantizer.fast_finetune(eval_loss, (quant_model, ft_loader, loss_fn))
        elif args.quant_mode == 'test':
            quantizer.load_ft_param()
    
    if args.evaluate:
        valid_loader = preprocessors.make_dataloader(data_dir=args.data_dir, 
                                                     data_file=args.calib_data, 
                                                     batch_size=args.batch_size)
        cr1 = CustomRunner(model=model, device=set_seed.DEVICE, 
                        input_key='features', input_target_key='targets', 
                        evaluate=True, loaders={'test': valid_loader})
        print('Evaluation completed!')
        print('Initial model results:')
        pprint.pprint(cr1.logs, width=5)
        
        if args.quant_mode != 'float':
            cr2 = CustomRunner(model=quant_model, device=set_seed.DEVICE, 
                            input_key='features', input_target_key='targets', 
                            evaluate=True, loaders={'test': valid_loader})
            print('Quantized model results:')
            pprint.pprint(cr2.logs, width=5)

    if args.quant_mode == 'calib':
        quantizer.export_quant_config()
    if args.deploy:
        quantizer.export_xmodel(deploy_check=True)


if __name__ == '__main__':
    model_name = os.path.splitext(args.model_name)[0]
    if args.quant_mode == 'float':
        mode_name = 'float evaluation'
    else:
        mode_name = 'quantization & optimization'
    running_mode = f'{model_name} {mode_name}'
    
    print("-------- Start {} test ".format(running_mode))
    quantization()
    print("-------- End of {} test ".format(running_mode))
