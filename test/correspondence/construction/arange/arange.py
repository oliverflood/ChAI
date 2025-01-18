import torch

torch.set_default_dtype(torch.float32)

def test(imports):
    print = imports['print_fn']

    a = torch.arange(6).reshape(2,3)
    print(a)

    b = torch.arange(24).reshape(2,3,4)
    print(b)

    c = torch.arange(10)
    print(c)

