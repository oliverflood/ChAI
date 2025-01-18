use Tensor;

var a = Tensor.arange(2,3);
writeln(a.degenerateFlatten());

var b = Tensor.arange(2,3,4);
writeln(b.degenerateFlatten());

var c = Tensor.arange(10);
writeln(c.degenerateFlatten());