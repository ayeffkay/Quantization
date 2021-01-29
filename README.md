# Project description
This is a toy project for testing pytorch version of Xilinx's Vitis-AI quantizer and compiler. Pytorch model description and its checkpoint are available at `model.py` and `checkpoints/ffn_state_dict.pth`. Model is quite simple and contains only three `torch.nn.Linear` layers and `relu`activations.

For detailed repo description see [files description](#Files-description).

# Files description

## Auxilary files
* generate_data.py -- generates synthetic data for training and evaluation
* set_seed.py -- fixing seeds for determitistic training
* runner.py -- class for pytorch model training and evaluation (based on Catalyst's Runner)
* preprocessors.py -- instances of torch Dataset and DataLoader
* losses.py -- modified MSE for convenient training


## Main files
* requirements.txt -- dependencies
* data -- data for training and evaluation (can be read via `numpy.load`)
* model.py -- toy pytorch model
* run.py -- training/evaluation of the pytorch model
* vai_q_pytorch_quant.py -- post-training quantization to int8 with `Vitis AI Quantizer` for pytorch 
* checkpoints -- pytorch model checkpoints
* quantize_result -- quantized model checkpoints after running `vai_q_pytorch_quant.py`


# Prerequisites
1. Install and run docker image for cpu from [Xilinx's github](https://github.com/Xilinx/Vitis-AI).
```
    git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI  
    cd Vitis-AI
    docker pull xilinx/vitis-ai-cpu:latest
    ./docker_run.sh xilinx/vitis-ai-cpu:latest
```
2. Activate conda environment for pytorch inside docker:
```
    conda activate vitis-ai-pytorch 
```

3. If you'll run any code from repo, install inside docker dependencies from `requirements.txt`:
```
    pip install -r requirements.txt
```

# Workflow
1. [Optional] Generate synthetic data (data files are already available at `data` folder)
```
    python generate_data.py --n_samples 10000 --output_dir data --seed 42
```
2. [Optional] Train and evaluate model (trained model checkpoint are already available at `checkpoint` folder)

```
    python run.py --data_dir data --lr 1e-1 --n_epochs 32 --batch_size 64 --train --input_size 2 --checkpoint_dir checkpoints
    python run.py --data_dir data --lr 1e-1 --n_epochs 32 --batch_size 64 --train --input_size 2 --checkpoint_dir checkpoints --evaluate --eval_data test.npy
```
or 

```
    python run.py --data_dir data --lr 1e-1 --n_epochs 32 --batch_size 64 --train --input_size 2 --checkpoint_dir checkpoints --evaluate --eval_data test.npy
```
3. [Optional] Quantize and deploy model with `vai_q_pytorch` tool (results are already available at `quantize_result` folder)

```
python vai_q_pytorch_quant.py --data_dir data --calib_data test.npy --input_size 2 --checkpoint_dir checkpoints --model_name ffn_state_dict.pth --batch_size 64 --quant_mode calib --evaluate

python vai_q_pytorch_quant.py --data_dir data --calib_data test.npy --input_size 2 --checkpoint_dir checkpoints --model_name ffn_state_dict.pth --batch_size 1 --subset_len 1 --quant_mode test --evaluate --deploy
```
4. Compile model with `vai_c_xir` for Alveo U250:

```
$> vai_c_xir -x quantize_result/FFN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json --net_name FFN_qd -o compile_output

**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[UNILOG][INFO] The compiler log will be dumped at "/tmp/vitis-ai-user/log/xcompiler-20210129-090633-271"
[UNILOG][INFO] Compiling subgraph : subgraph_FFN__FFN_Linear_l1__10
 ############################################# 
 ######  Hyper Graph Construction 
 ############################################# 
 ############################################# 
 ######  Hyper Graph Construction
 ############################################# 
Floyd & Warshall
BFS
 ############################################# 
 ######  Parameters Assimilation: # 9 
 ############################################# 
 ############################################# 
 ######  Assimilating Fix Neurons: # 3 
 ############################################# 
 ############################################# 
 ######  Assimilating Relu: # 2 
 ############################################# 
 ############################################# 
 ######  Assimilating LeakyRelu: # 0 
 ############################################# 
 ############################################# 
 ######  I like VALID more than SAME
 ############################################# 
 ############################################# 
 ######  I like VALID more than SAME: # 0 
 ############################################# 
 ############################################# 
 ######  Assimilating Padding:# 0 
 ############################################# 
 ############################################# 
 ######  CPU nodes Must Go
 ############################################# 
Inputs ['FFN__input_0_fix']
Outputs ['FFN__FFN_Linear_l3__20']
FPGA True: data       FFN__input_0_fix  
FPGA True: matmul     FFN__FFN_Linear_l1__10  
FPGA True: matmul     FFN__FFN_Linear_l2__15  
FPGA True: matmul     FFN__FFN_Linear_l3__20  
delete these dict_keys(['FFN__input_0_fix'])
{'FFN__FFN_Linear_l3__20': Name FFN__FFN_Linear_l3__20 Type matmul Composed [] Inputs ['FFN__FFN_Linear_l2__15'] 
}
Schedule boost
0 data FFN__input_0_fix False 1
1 matmul FFN__FFN_Linear_l1__10 True 1
2 matmul FFN__FFN_Linear_l2__15 True 1
3 matmul FFN__FFN_Linear_l3__20 True 1
Outputs ['FFN__FFN_Linear_l3__20']
Inputs  ['FFN__input_0_fix']
Floyd & Warshall
BFS
 ############################################# 
 ######  Avg Pool -> Conv
 ############################################# 
 ############################################# 
 ######  Inner Products -> Conv
 ############################################# 
 MULT FFN__FFN_Linear_l1__10 -> CONV
 MULT FFN__FFN_Linear_l2__15 -> CONV
 MULT FFN__FFN_Linear_l3__20 -> CONV
 ############################################# 
 ######  Scale -> Conv
 ############################################# 
 ############################################# 
 ######   Concat of concat
 ############################################# 
Floyd & Warshall
BFS
 ############################################# 
 ######  topological schedule BFS
 ############################################# 
 ############################################# 
 ######  WEIGHT & BIAS into Tensors
 ############################################# 
 ############################################# 
 ######  topological DFS
 ############################################# 
DFS_t FFN__input_0_fix
 ############################################# 
 ######  TFS
 ############################################# 
 ############################################# 
 ######  INC
 ############################################# 
INC
 ############################################# 
 ######  Singleton
 ############################################# 
  0 data       FFN__input_0_fix Ops 0 Shape [1, 2]  IN [] OUT ['FFN__FFN_Linear_l1__10']
  1 matmul     FFN__FFN_Linear_l1__10 Ops 0 Shape [1, 6]  IN ['FFN__input_0_fix'] OUT ['FFN__FFN_Linear_l2__15']
  2 matmul     FFN__FFN_Linear_l2__15 Ops 0 Shape [1, 4]  IN ['FFN__FFN_Linear_l1__10'] OUT ['FFN__FFN_Linear_l3__20']
  3 matmul     FFN__FFN_Linear_l3__20 Ops 0 Shape [1, 1]  IN ['FFN__FFN_Linear_l2__15'] OUT []
 ############################################# 
 ######  Given a Graph and Schedule boost : We crete Live Tensor
 ############################################# 
 ############################################# 
 ######  Reset Live Structure
 ############################################# 
 ############################################# 
 ######  Attempting Code Generation boost
 ############################################# 
 ############################################# 
 ######  Element Wise: reuse one of the operands
 ############################################# 
 ############################################# 
 ######  Concatenation: I love concatenation memory reuse
 ############################################# 
 ############################################# 
 ######  Memory Management given a Schedule
 ############################################# 
> /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/HwAbstraction/gen_param_ddr_demo.py(123)bias_data_int()
-> shift_bias = pos_i + pos_w - pos_b #-1
(Pdb) --KeyboardInterrupt--
(Pdb) --KeyboardInterrupt--
(Pdb) 
terminate called after throwing an instance of 'pybind11::error_already_set'
  what():  BdbQuit: 

At:
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/bdb.py(70): dispatch_line
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/bdb.py(51): trace_dispatch
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/HwAbstraction/gen_param_ddr_demo.py(123): bias_data_int
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/HwAbstraction/gen_param_ddr_demo.py(49): reshape_bias_to_ddr
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/Scheduler/memory.py(819): from_par_to_string
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/Scheduler/memory.py(930): initialize
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/Scheduler/memory.py(879): __init__
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/HwAbstraction/code_generation.py(302): my_simplified_code
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/HwAbstraction/code_generation.py(511): explore_code_generation
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/HwAbstraction/code_generation.py(799): compile_sc_dpuv3int8
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/compiler_func.py(22): compile_sc_dpuv3int8_default
  /opt/vitis_ai/conda/envs/vitis-ai-pytorch/lib/python3.6/site-packages/SC/compiler_func.py(26): compile_subgraph_inplace

Aborted (core dumped)

```

or

```
$> vai_c_xir -x quantize_result/FFN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCADX8G/ALVEO/arch.json --net_name FFN_qd -o compile_output

**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
ERROR: NO FRONT END SPECIFIED

```


