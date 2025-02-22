use Tensor;

var a = ndarray.arange(2,3);

writeln(a);

writeln(a.sum(0));

writeln(a.sum(1));

writeln(a.mean(0));
writeln(a.mean(1));