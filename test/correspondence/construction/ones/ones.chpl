use Tensor;

var a = Tensor.ones(2,3);
writeln(a.degenerateFlatten());

var b = Tensor.ones(2,3,4);
writeln(b.degenerateFlatten());

var c = Tensor.ones(10);
writeln(c.degenerateFlatten());