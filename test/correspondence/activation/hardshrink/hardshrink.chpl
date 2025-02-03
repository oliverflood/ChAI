use Tensor;

var a = Tensor.zeros(2,3).hardshrink();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).hardshrink();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).hardshrink();
writeln(c.degenerateFlatten());

// same values with alpha = -0.001
a = Tensor.zeros(2,3).hardshrink(alpha=-0.001);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).hardshrink(alpha=-0.001);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).hardshrink(alpha=-0.001);
writeln(c.degenerateFlatten());

// same values with alpha = 10.0
a = Tensor.zeros(2,3).hardshrink(alpha=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).hardshrink(alpha=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).hardshrink(alpha=10.0);
writeln(c.degenerateFlatten());