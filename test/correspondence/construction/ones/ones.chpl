use Tensor;

var a = Tensor.ones(2,3);
writeln(a.forceRank(2).array.degenerateFlatten());

var b = Tensor.ones(2,3,4);
writeln(b.forceRank(3).array.degenerateFlatten());

var c = Tensor.ones(10);
writeln(c.forceRank(1).array.degenerateFlatten());