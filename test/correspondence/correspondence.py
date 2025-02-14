import os
import sys
from pathlib import Path
import torch

from decimal import Decimal
import decimal


import argparse

correspondence_dir = Path(__file__).parent # Path('.')
chai_dir = correspondence_dir.parent.parent
chai_test_dir = chai_dir / 'test'
chai_lib_dir = chai_dir / 'lib'
chai_py_path = chai_lib_dir / 'chai.py'

parser = argparse.ArgumentParser(description='Run correspondence tests.')
# Directories and paths
parser.add_argument('--correspondence-dir', type=Path, default=correspondence_dir, help='Path to correspondence directory.')
parser.add_argument('--chai-dir', type=Path, default=chai_dir, help='Path to Chai directory.')
parser.add_argument('--chai-test-dir', type=Path, default=chai_test_dir, help='Path to Chai test directory.')
parser.add_argument('--chai-lib-dir', type=Path, default=chai_lib_dir, help='Path to Chai lib directory.')
parser.add_argument('--chai-py-path', type=Path, default=chai_py_path, help='Path to Chai Python module.')

parser.add_argument('--test-only', type=Path, nargs='*', help='Run a specific test(s).')

args = parser.parse_args()

# Update directories if provided as arguments
correspondence_dir = args.correspondence_dir
chai_dir = args.chai_dir
chai_test_dir = args.chai_test_dir
chai_lib_dir = args.chai_lib_dir
chai_py_path = args.chai_py_path


# Add chai lib to path
try:
    sys.path.append(str(chai_lib_dir))
except Exception as e:
    print('Failed to add chai lib to path.')
    print(e)
    print('Exiting...')
    sys.exit(1)

# Import chai module
import chai


test_dirs_to_run = [correspondence_dir / x for x in args.test_only] if args.test_only else None

# Validate test directories to run
if test_dirs_to_run:
    for test_dir in test_dirs_to_run:
        if not test_dir.is_dir():
            raise NotADirectoryError(f'Invalid test directory: {test_dir}')

def should_run_test(test_dir):
    if not test_dirs_to_run:
        return True
    return test_dir in test_dirs_to_run



correspondence_test_types = {}
for x in correspondence_dir.iterdir():
    if x.is_dir():
        correspondence_test_types[x.name] = x


tests = []

def test_dir_source_paths(name,test_dir):
    chapel_test_file_path = test_dir / f'{name}.chpl'
    python_test_file_path = test_dir / f'{name}.py'
    return (chapel_test_file_path,python_test_file_path)

def is_test_dir(name,test_dir):
    chapel_test_file_path,python_test_file_path = test_dir_source_paths(name,test_dir)
    return (test_dir.is_dir() 
            and chapel_test_file_path.is_file() 
            and python_test_file_path.is_file())


for test_type, test_type_dir in correspondence_test_types.items():
    test_dirs = {x.name : x for x in test_type_dir.iterdir() if x.is_dir()}
    for test_name, test_dir in test_dirs.items():
        if not should_run_test(test_dir):
            continue
        if is_test_dir(test_name,test_dir):
            test_info = {
                'type': test_type,
                'name': test_name,
                'relative_path': test_dir.relative_to(correspondence_dir),
                'test_path': test_dir.relative_to(chai_dir),
                'absolute_path': test_dir
            }
            tests.append(test_info)
            print('🌱\t', test_info['relative_path'])


def compile_chapel(test_name,test_path,chai_path):
    assert test_name == test_path.name
    chapel_test_path = test_path / f'{test_name}.chpl'
    test_dir = chapel_test_path.parent
    chai_lib_path = chai_path / 'lib'
    compile_cmd = f'chpl {chapel_test_path} -M {chai_lib_path} -o {test_dir / test_name}'
    # os.system(compile_cmd)
    import subprocess
    results = subprocess.run(compile_cmd,capture_output=True,shell=True,text=True) #stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if results.returncode != 0:
        # print('Failed to compile', chapel_test_path)
        # print(results.stdout)
        # print(results.stderr)
        raise Exception(f'Failed to compile {test_name}.')

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
    printer = Printer()

    sys.path.append(str(test_path))
    py_test_module = __import__(test_name)
    py_test_module.test({'print_fn': printer.print})
        
    return str(printer)

success_emoji = '✅'
failure_emoji = '❌'

def print_failure(test,python_output,chapel_output):
    print('-------- failed --------')
    print('Python output:')
    print(python_output)
    print('Chapel output:')
    print(chapel_output)
    print(failure_emoji, test['test_path'])
    raise Exception(f'Failed test {test["name"]}.')

def print_success(test):
    print(success_emoji, test['test_path'])

def tokenize_output(output):
    import re
    return re.findall(r'\S+', output)

def parse_output(output_tokens):
    nums = []
    for t in output_tokens:
        d = Decimal(t)
        f = "%f" % d
        nums.append(Decimal(f))
    return nums


failed_python_tests = []
failed_compilation_tests = []
failed_tests = []

for test in tests:
    test_name = test['name']
    test_type = test['type']
    test_path = test['absolute_path']

    # acceptable_tests = [
    #     'arange',
    #     'ones',
    #     'relu'
    # ]
    # if test_name not in acceptable_tests:
    #     continue

    # tests_to_skip = [
    #     'rrelu',
    #     'hardtanh'
    # ]
    # if test_name in tests_to_skip:
    #     continue

    try:
        python_output = run_python_test(test_name,test_path)
    except Exception as e:
        print('🐍', test['test_path'])
        failed_python_tests.append(test['name'])
        continue
    # print(f'Compiling {test_name}...')
    try:
        compile_chapel(test_name,test_path,chai_dir)
    except Exception as e:
        print('⛔', test['test_path'])
        failed_compilation_tests.append(test['name'])
        continue

    # print(f'Running {test_name}...')
    chapel_output = run_chapel_test(test_name,test_path)

    python_output_tokens = tokenize_output(python_output)
    chapel_output_tokens = tokenize_output(chapel_output)

    assert len(python_output_tokens) == len(chapel_output_tokens)

    output_size = len(python_output_tokens)

    python_results = parse_output(python_output_tokens)
    chapel_results = parse_output(chapel_output_tokens)


    failed = False

    for i in range(output_size):
        pr = python_results[i]
        cr = chapel_results[i]
        if pr != cr:
            failed = True

    if failed:
        print(failure_emoji, test['test_path'])
        failed_tests.append(test['name'])
    else:
        print(success_emoji, test['test_path'])

    continue

    if failed:
        for i in range(output_size):
            pr = python_results[i]
            cr = chapel_results[i]
            pt = python_output_tokens[i]
            ct = chapel_output_tokens[i]
            if pr != cr:
                print(f'❌ {pr} != {cr}, {pt} != {ct}')
            else:
                print(f'✅ {pr} == {cr}, {pt} == {ct}')
        print(failure_emoji, test['test_path'])
        sys.exit(0)
    else:
        print(success_emoji, test['test_path'])

    # if chapel_output != python_output:
    #     print_failure(test,python_output,chapel_output)
    # else:
    #     print_success(test)

    python_test_path = test['absolute_path'] / f'{test_name}.py'

print('Failed Chapel compilations tests:', failed_compilation_tests)
print('Failed Python tests:', failed_python_tests)
print('Tests to fix:', failed_tests)