import torch
import torch.nn.functional as F

def test(imports):
    print = imports['print_fn']

    a = F.rrelu(torch.zeros(2,3))
    print(a)

    b = F.rrelu(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = F.rrelu(torch.zeros(10) + 40.0)
    print(c)

    # same values with lower = -0.001
    a = F.rrelu(torch.zeros(2,3),lower=-10.0)
    print(a)

    b = F.rrelu(torch.zeros(2,3,4) - 60.0,lower=-10.0)
    print(b)

    c = F.rrelu(torch.zeros(10) + 40.0,lower=-10.0)
    print(c)
    
    # same values with upper = 10.0
    a = F.rrelu(torch.zeros(2,3),upper=10.0)
    print(a)

    b = F.rrelu(torch.zeros(2,3,4) - 60.0,upper=10.0)
    print(b)

    c = F.rrelu(torch.zeros(10) + 40.0,upper=10.0)
    print(c)
    
    # same values with lower = -50.0, upper = 30.0
    a = F.rrelu(torch.zeros(2,3),lower=-50.0, upper=30.0)
    print(a)

    b = F.rrelu(torch.zeros(2,3,4) - 60.0,lower=-50.0, upper=30.0)
    print(b)

    c = F.rrelu(torch.zeros(10) + 40.0,lower=-50.0, upper=30.0)
    print(c)