use Tensor;

var a = Tensor.zeros(2,3).sigmoid();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).sigmoid();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).sigmoid();
writeln(c.degenerateFlatten());