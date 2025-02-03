use Tensor;

var a = Tensor.zeros(2,3).softsign();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 60.0).softsign();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).softsign();
writeln(c.degenerateFlatten());