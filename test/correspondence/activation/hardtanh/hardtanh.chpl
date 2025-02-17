use Tensor;

var a = Tensor.zeros(2,3).hardtanh();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).hardtanh();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).hardtanh();
Testing.numericPrint(c);

// same values with minVal = -0.001
a = Tensor.zeros(2,3).hardtanh(minVal=-0.001);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardtanh(minVal=-0.001);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardtanh(minVal=-0.001);
Testing.numericPrint(c);

// same values with maxVal = 10.0
a = Tensor.zeros(2,3).hardtanh(maxVal=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardtanh(maxVal=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardtanh(maxVal=10.0);
Testing.numericPrint(c);

// same values with minVal = -50.0, maxVal = 30.0
a = Tensor.zeros(2,3).hardtanh(minVal = -50.0, maxVal=30.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardtanh(minVal=-50.0, maxVal=30.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardtanh(minVal=-50.0, maxVal=30.0);
Testing.numericPrint(c);