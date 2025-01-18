import torch

def test(imports):
    print = imports['print_fn']

    a = torch.ones(2,3)
    print(a)

    b = torch.ones(2,3,4)
    print(b)

    c = torch.ones(10)
    print(c)
