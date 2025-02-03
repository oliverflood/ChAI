import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.RReLU()(torch.zeros(2,3))
    print(a)

    b = torch.nn.RReLU()(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.RReLU()(torch.zeros(10) + 4.0)
    print(c)
    
    # same values with lower = -0.001
    a = torch.nn.RReLU(lower=-10.0, )(torch.zeros(2,3))
    print(a)

    b = torch.nn.RReLU(lower=-10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.RReLU(lower=-10.0)(torch.zeros(10) + 4.0)
    print(c)
    
    # same values with upper = 10.0
    a = torch.nn.RReLU(upper=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.RReLU(upper=10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.RReLU(upper=10.0)(torch.zeros(10) + 4.0)
    print(c)
    
    # same values with lower = -50.0, upper = 30.0
    a = torch.nn.RReLU(lower=-50.0, upper=30.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.RReLU(lower=-50.0, upper=30.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.RReLU(lower=-50.0, upper=30.0)(torch.zeros(10) + 4.0)
    print(c)