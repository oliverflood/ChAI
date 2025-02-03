import torch

def test(imports):
    print = imports['print_fn']

    a = torch.nn.Threshold()(torch.zeros(2,3))
    print(a)

    b = torch.nn.Threshold()(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Threshold()(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with threshold = -10.0
    a = torch.nn.Threshold(threshold=-10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Threshold(threshold=-10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Threshold(threshold=-10.0)(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with value = 10.0
    a = torch.nn.Threshold(value=10.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Threshold(value=10.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Threshold(value=10.0)(torch.zeros(10) + 40.0)
    print(c)
    
    # same values with threshold = -50.0, value = 30.0
    a = torch.nn.Threshold(threshold=-50.0, value=30.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Threshold(threshold=-50.0, value=30.0)(torch.zeros(2,3,4) - 1.0)
    print(b)

    c = torch.nn.Threshold(threshold=-50.0, value=30.0)(torch.zeros(10) + 40.0)
    print(c)