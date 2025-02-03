use Tensor;

var a = Tensor.zeros(2,3).silu();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 60.0).silu();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).silu();
writeln(c.degenerateFlatten());