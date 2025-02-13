use Tensor;

var a = Tensor.zeros(2,3).sigmoid();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).sigmoid();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).sigmoid();
Testing.numericPrint(c);