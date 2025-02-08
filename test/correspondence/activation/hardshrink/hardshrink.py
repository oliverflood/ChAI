import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Hardshrink()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardshrink()(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Hardshrink()(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with alpha = -0.001
    a = torch.nn.Hardshrink(l=-0.001)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardshrink(l=-0.001)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Hardshrink(l=-.001)(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with alpha = 10.0
    a = torch.nn.Hardshrink(l=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardshrink(l=10.0)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Hardshrink(l=10.0)(torch.zeros(10) + 40.0)
    print(c)