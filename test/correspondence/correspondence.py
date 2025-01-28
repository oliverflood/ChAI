import os
import sys
from pathlib import Path
import torch

correspondence_dir = Path(__file__).parent # Path('.')
chai_dir = correspondence_dir.parent.parent
chai_test_dir = chai_dir / 'test'
chai_lib_dir = chai_dir / 'lib'
chai_py_path = chai_lib_dir / 'chai.py'

# import the chai module
sys.path.append(str(chai_lib_dir))
import chai

correspondence_test_types = {x.name : x for x in correspondence_dir.iterdir() if x.is_dir()}

tests = []

def test_dir_paths(name,path):
    chapel_test_file_dir = path / f'{name}.chpl'
    python_test_file_dir = path / f'{name}.py'
    return (chapel_test_file_dir,python_test_file_dir)

def is_test_dir(name,path):
    chapel_test_file_dir,python_test_file_dir = test_dir_paths(name,path)
    return path.is_dir() and chapel_test_file_dir.is_file() and python_test_file_dir.is_file()    


for test_type, test_type_dir in correspondence_test_types.items():
    test_dirs = {x.name : x for x in test_type_dir.iterdir() if x.is_dir()}
    for test_name, test_path in test_dirs.items():
        if is_test_dir(test_name,test_path):
            test_info = {
                'type': test_type,
                'name': test_name,
                'relative_path': test_path.relative_to(correspondence_dir),
                'test_path': test_path.relative_to(chai_dir),
                'absolute_path': test_path
            }
            tests.append(test_info)
            print('[found test]\t', test_info['name'], '\tin\t', test_info['test_path'])


def compile_chapel(test_name,test_path,chai_path):
    assert test_name == test_path.name
    chapel_test_path = test_path / f'{test_name}.chpl'
    test_dir = chapel_test_path.parent
    chai_lib_path = chai_path / 'lib'
    os.system(f'chpl {chapel_test_path} -M {chai_lib_path} -o {test_dir / test_name}')


def run_chapel_test(test_name,test_path):
    assert test_name == test_path.name
    test_exec = test_path / test_name
    import subprocess
    output = subprocess.getoutput(f'{test_exec}')
    return output

class Printer(object):
    def __init__(self):
        self.data = []
    
    def print(self,*args):
        self.data.append(args)

    def __str__(self):
        def process(x):
            if isinstance(x,torch.Tensor):
                dtype = torch.get_default_dtype() # TODO: This may need to change for global coherency.
                xs = x.to(dtype).flatten().tolist()
                xss = ' '.join([str(x) for x in xs])
                return xss
            else:
                return str(x)
            
        spacer = ' ' # '\t' # TODO: This may need to change!
        lines = []
        for d in self.data:
            if isinstance(d,tuple):
                lines.append(spacer.join([process(x) for x in d]))
            elif isinstance(d,list):
                lines.append(spacer.join([process(x) for x in d]))
            else:
                lines.append(process(d))
        return '\n'.join(lines)

def run_python_test(test_name,test_path):
    assert test_name == test_path.name
    python_test_path = test_path / f'{test_name}.py'

    sys.path.append(str(test_path))
    py_test_module = __import__(test_name)

    printer = Printer()
    py_test_module.test({'print_fn': printer.print})
    return str(printer)

for test in tests:
    test_name = test['name']
    test_type = test['type']
    test_path = test['absolute_path']

    python_output = run_python_test(test_name,test_path)

    # print(f'Compiling {test_name}...')
    compile_chapel(test_name,test_path,chai_dir)

    # print(f'Running {test_name}...')
    chapel_output = run_chapel_test(test_name,test_path)
    if chapel_output != python_output:
        print('-------- failed --------')
        print('Python output:')
        print(python_output)
        print('Chapel output:')
        print(chapel_output)
    else:
        print('[passed]', test_name)

    # python_test_path = test['absolute_path'] / f'{test_name}.py'

