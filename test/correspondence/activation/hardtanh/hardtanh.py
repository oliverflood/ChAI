import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Hardtanh()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardtanh()(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Hardtanh()(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with min_val = -0.001
    a = torch.nn.Hardtanh(min_val=-10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardtanh(min_val=-10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Hardtanh(min_val=-10.0)(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with max_val = 10.0
    a = torch.nn.Hardtanh(max_val=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardtanh(max_val=10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Hardtanh(max_val=10.0)(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with min_val = -50.0, max_val = 30.0
    a = torch.nn.Hardtanh(min_val=-50.0, max_val=30.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Hardtanh(min_val=-50.0, max_val=30.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Hardtanh(min_val=-50.0, max_val=30.0)(torch.zeros(10) + 40.0)
    print(c)