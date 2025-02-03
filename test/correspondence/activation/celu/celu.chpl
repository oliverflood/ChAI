use Tensor;

var a = Tensor.zeros(2,3).celu();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).celu();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).celu();
writeln(c.degenerateFlatten());