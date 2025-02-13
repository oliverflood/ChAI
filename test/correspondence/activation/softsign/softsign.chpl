use Tensor;

var a = Tensor.zeros(2,3).softsign();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).softsign();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).softsign();
Testing.numericPrint(c);