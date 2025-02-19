use Tensor;

var a = Tensor.zeros(2,3).celu(alpha=-0.001);
Testing.numericPrint(a);