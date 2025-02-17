import torch

def test(imports):
    print = imports['print_fn']

    # threshold has no default values
    a = torch.nn.Threshold(threshold=-50.0, value=30.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Threshold(threshold=-50.0, value=30.0)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Threshold(threshold=-50.0, value=30.0)(torch.zeros(10) + 40.0)
    print(c)
    
    a = torch.nn.Threshold(threshold=5.0, value=0.0)(torch.zeros(2,3))
    print(a)

    b = torch.nn.Threshold(threshold=-5.0, value=0.0)(torch.zeros(2,3,4) - 60.0)
    print(b)

    c = torch.nn.Threshold(threshold=-5.0, value=0.0)(torch.zeros(10) + 40.0)
    print(c)
    
    a = torch.nn.Threshold(threshold=2.0, value=2.0)(torch.zeros(10) + 40.0)
    print(a)