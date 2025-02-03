use Tensor;

var a = Tensor.zeros(2,3).relu6();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).relu6();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).relu6();
writeln(c.degenerateFlatten());