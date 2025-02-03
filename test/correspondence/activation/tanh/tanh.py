import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Tanh()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Tanh()(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Tanh()(torch.zeros(10) + 40.0)
    print(c)