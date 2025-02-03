import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.ELU()(torch.zeros(2,3))
    print(a)

    b = torch.nn.ELU()(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.ELU()(torch.zeros(10) + 4.0)
    print(c)
    
    # same values with alpha = -0.001
    a = torch.nn.ELU(alpha=-0.001)(torch.zeros(2,3))
    print(a)

    b = torch.nn.ELU(alpha=-0.001)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.ELU(alpha=-.001)(torch.zeros(10) + 4.0)
    print(c)
    
    # same values with alpha = 10.0
    a = torch.nn.ELU(alpha=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.ELU(alpha=10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.ELU(alpha=10.0)(torch.zeros(10) + 4.0)
    print(c)