use Tensor;

var a = Tensor.zeros(2,3).hardtanh();
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).hardtanh();
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).hardtanh();
Testing.numericPrint(c);

// same values with min_val = -0.001
a = Tensor.zeros(2,3).hardtanh(min_val=-0.001);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardtanh(min_val=-0.001);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardtanh(min_val=-0.001);
Testing.numericPrint(c);

// same values with max_val = 10.0
a = Tensor.zeros(2,3).hardtanh(max_val=10.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardtanh(max_val=10.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardtanh(max_val=10.0);
Testing.numericPrint(c);

// same values with min_val = -50.0, max_val = 30.0
a = Tensor.zeros(2,3).hardtanh(min_val = -50.0, max_val=30.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).hardtanh(min_val=-50.0, max_val=30.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).hardtanh(min_val=-50.0, max_val=30.0);
Testing.numericPrint(c);