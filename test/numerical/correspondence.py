import torch
import torch.nn.functional as F

class ChaiExpr:
    def __init__(self,name,children):
        self.name = name
        self.children = children

    def compile_fn(self,*args):
        pass

    def eval_fn(self,*args):
        pass
    
    def compile(self):
        return self.compile_fn(*[child.compile() for child in self.children])

class Ones(ChaiExpr):
    def __init__(self,shape):
        super().__init__('ones',[])
        self.shape = shape

    def compile_fn(self):
        return 'tensor.ones' + str(self.shape)

    def eval_fn(self):
        return torch.ones(*self.shape)
    
