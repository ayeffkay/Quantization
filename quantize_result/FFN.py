# GENETARED BY NNDCT, DO NOT EDIT!

import torch
import pytorch_nndct as py_nndct
class FFN(torch.nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #FFN::input_0
        self.module_1 = py_nndct.nn.Linear(in_features=2, out_features=6, bias=True) #FFN::FFN/Linear[l1]/10
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #FFN::FFN/input.2
        self.module_3 = py_nndct.nn.Linear(in_features=6, out_features=4, bias=True) #FFN::FFN/Linear[l2]/15
        self.module_4 = py_nndct.nn.ReLU(inplace=False) #FFN::FFN/input
        self.module_5 = py_nndct.nn.Linear(in_features=4, out_features=1, bias=True) #FFN::FFN/Linear[l3]/20

    def forward(self, *args):
        self.output_module_0 = self.module_0(input=args[0])
        self.output_module_1 = self.module_1(self.output_module_0)
        self.output_module_2 = self.module_2(self.output_module_1)
        self.output_module_3 = self.module_3(self.output_module_2)
        self.output_module_4 = self.module_4(self.output_module_3)
        self.output_module_5 = self.module_5(self.output_module_4)
        return self.output_module_5
