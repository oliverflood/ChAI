import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Softshrink()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softshrink()(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Softshrink()(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with alpha = 10.0
    a = torch.nn.Softshrink(lambd=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Softshrink(lambd=10.0)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Softshrink(lambd=10.0)(torch.zeros(10) + 40.0)
    print(c)