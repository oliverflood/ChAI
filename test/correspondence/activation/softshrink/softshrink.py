import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Softshrink()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softshrink()(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Softshrink()(torch.zeros(10) + 4.0)
    print(c)
    
    # same values with alpha = 10.0
    a = torch.nn.Softshrink(l=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softshrink(l=10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Softshrink(l=10.0)(torch.zeros(10) + 4.0)
    print(c)