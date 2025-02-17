use Tensor;

var a = Tensor.zeros(2,3).selu();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).selu();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).selu();
Testing.numericPrint(c);