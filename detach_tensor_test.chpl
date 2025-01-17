use Tensor;

use Autograd;


var a: tensor(2,real) = tensor.arange(3,3);

writeln(a);


var b: tensor(2,real) = tensor.arange(3,3) * -0.1;
writeln(b);

