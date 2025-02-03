use Tensor;

var a = Tensor.zeros(2,3).logsigmoid();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).logsigmoid();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).logsigmoid();
writeln(c.degenerateFlatten());