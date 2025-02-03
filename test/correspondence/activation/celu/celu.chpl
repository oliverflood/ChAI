use Tensor;

var a = Tensor.zeros(2,3).celu();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).celu();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).celu();
writeln(c.degenerateFlatten());

// same values with alpha = -0.001
a = Tensor.zeros(2,3).celu(alpha=-0.001);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).celu(alpha=-0.001);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).celu(alpha=-0.001);
writeln(c.degenerateFlatten());

// same values with alpha = 10.0
a = Tensor.zeros(2,3).celu(alpha=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).celu(alpha=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).celu(alpha=10.0);
writeln(c.degenerateFlatten());