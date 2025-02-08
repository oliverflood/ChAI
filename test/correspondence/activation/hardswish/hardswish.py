import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Hardswish()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardswish()(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Hardswish()(torch.zeros(10) + 40.0)
    print(c)