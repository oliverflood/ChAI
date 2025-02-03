use Tensor;

var a = Tensor.zeros(2,3).hardtanh();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).hardtanh();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).hardtanh();
writeln(c.degenerateFlatten());

// same values with min_val = -0.001
a = Tensor.zeros(2,3).hardtanh(min_val=-0.001);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).hardtanh(min_val=-0.001);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).hardtanh(min_val=-0.001);
writeln(c.degenerateFlatten());

// same values with max_val = 10.0
a = Tensor.zeros(2,3).hardtanh(max_val=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).hardtanh(max_val=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).hardtanh(max_val=10.0);
writeln(c.degenerateFlatten());

// same values with min_val = -50.0, max_val = 30.0
a = Tensor.zeros(2,3).hardtanh(min_val = -50.0, max_val=30.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).hardtanh(min_val=-50.0, max_val=30.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).hardtanh(min_val=-50.0, max_val=30.0);
writeln(c.degenerateFlatten());