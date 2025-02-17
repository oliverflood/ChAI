use Tensor;

var a = Tensor.zeros(2,3).softshrink();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).softshrink();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).softshrink();
Testing.numericPrint(c);

// same values with alpha = 10.0
a = Tensor.zeros(2,3).softshrink(alpha=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).softshrink(alpha=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).softshrink(alpha=10.0);
Testing.numericPrint(c);