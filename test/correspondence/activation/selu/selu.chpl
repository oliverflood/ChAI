use Tensor;

var a = Tensor.zeros(2,3).selu();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).selu();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).selu();
writeln(c.degenerateFlatten());