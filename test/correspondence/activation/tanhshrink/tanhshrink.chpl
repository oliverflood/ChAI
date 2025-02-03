use Tensor;

var a = Tensor.zeros(2,3).tanhshrink();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).tanhshrink();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).tanhshrink();
writeln(c.degenerateFlatten());