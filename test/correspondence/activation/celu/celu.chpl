use Tensor;

var a = Tensor.zeros(2,3).celu();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).celu();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).celu();
Testing.numericPrint(c);

// same values with alpha = -0.001
a = Tensor.zeros(2,3).celu(alpha=-0.001);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).celu(alpha=-0.001);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).celu(alpha=-0.001);
Testing.numericPrint(c);

// same values with alpha = 10.0
a = Tensor.zeros(2,3).celu(alpha=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).celu(alpha=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).celu(alpha=10.0);
Testing.numericPrint(c);