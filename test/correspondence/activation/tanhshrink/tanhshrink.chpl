use Tensor;

var a = Tensor.zeros(2,3).tanhshrink();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).tanhshrink();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).tanhshrink();
Testing.numericPrint(c);