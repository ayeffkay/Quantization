# Usage notes

* If you'll run any script from repo, first install dependencies from `requirements.txt`:

```
pip install -r requirements.txt
```

* For compilation only use folder `quantize_result` and run:

```
vai_c_xir -x quantize_result/FFN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCADF8H/U250/arch.json --net_name FFN_qd -o compile_output
```

or (both options don't work)

```
vai_c_xir -x quantize_result/FFN_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCADX8G/ALVEO/arch.json --net_name FFN_qd -o compile_output
```
First throws into `pdb`, second produces `ERROR: NO FRONT END SPECIFIED`.

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
