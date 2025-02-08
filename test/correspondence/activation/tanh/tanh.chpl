use Tensor;

var a = Tensor.zeros(2,3).tanh();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 60.0).tanh();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).tanh();
writeln(c.degenerateFlatten());