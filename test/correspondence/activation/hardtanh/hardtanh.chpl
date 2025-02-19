use Tensor;

var a = Tensor.zeros(2,3).hardTanh();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).hardTanh();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).hardTanh();
Testing.numericPrint(c);

// same values with minVal = -0.001
a = Tensor.zeros(2,3).hardTanh(minVal=-0.001);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardTanh(minVal=-10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardTanh(minVal=-10.0);
Testing.numericPrint(c);

// same values with maxVal = 10.0
a = Tensor.zeros(2,3).hardTanh(maxVal=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardTanh(maxVal=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardTanh(maxVal=10.0);
Testing.numericPrint(c);

// same values with minVal = -50.0, maxVal = 30.0
a = Tensor.zeros(2,3).hardTanh(minVal = -50.0, maxVal=30.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardTanh(minVal=-50.0, maxVal=30.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardTanh(minVal=-50.0, maxVal=30.0);
Testing.numericPrint(c);