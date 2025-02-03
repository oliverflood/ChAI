use Tensor;

var a = Tensor.zeros(2,3).mish();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 60.0).mish();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).mish();
writeln(c.degenerateFlatten());