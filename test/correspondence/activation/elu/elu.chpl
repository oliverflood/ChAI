use Tensor;

var a = Tensor.zeros(2,3).elu();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).elu();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).elu();
Testing.numericPrint(c);

// same values with alpha = -0.001
a = Tensor.zeros(2,3).elu(alpha=-0.001);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).elu(alpha=-0.001);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).elu(alpha=-0.001);
Testing.numericPrint(c);

// same values with alpha = 10.0
a = Tensor.zeros(2,3).elu(alpha=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).elu(alpha=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).elu(alpha=10.0);
Testing.numericPrint(c);