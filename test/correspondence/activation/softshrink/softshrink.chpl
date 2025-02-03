use Tensor;

var a = Tensor.zeros(2,3).softshrink();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).softshrink();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).softshrink();
writeln(c.degenerateFlatten());

// same values with alpha = 10.0
a = Tensor.zeros(2,3).softshrink(alpha=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).softshrink(alpha=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).softshrink(alpha=10.0);
writeln(c.degenerateFlatten());