use Tensor;

var a = Tensor.zeros(2,3).threshold();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).threshold();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).threshold();
Testing.numericPrint(c);

// same values with threshold = -10.0
a = Tensor.zeros(2,3).threshold(threshold=-10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).threshold(threshold=-10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).threshold(threshold=-10.0);
Testing.numericPrint(c);

// same values with value = 10.0
a = Tensor.zeros(2,3).threshold(value=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).threshold(value=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).threshold(value=10.0);
Testing.numericPrint(c);

// same values with threshold = -50.0, value = 30.0
a = Tensor.zeros(2,3).threshold(threshold = -50.0, value=30.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).threshold(threshold=-50.0, value=30.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).threshold(threshold=-50.0, value=30.0);
Testing.numericPrint(c);