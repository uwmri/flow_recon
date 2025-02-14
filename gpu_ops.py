import torch
import sigpy as sp

__all__ = ['array_to_gpu',]

def array_to_gpu( a , device=sp.Device(0)):
    #a = torch.tensor(a)
    #a = a.cuda()
    #a = sp.from_pytorch(a)
    return sp.to_device(a, device)
