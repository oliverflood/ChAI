use Tensor;

var a = Tensor.zeros(2,3).softsign();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).softsign();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).softsign();
writeln(c.degenerateFlatten());