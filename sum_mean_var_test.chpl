use Tensor;

var a = ndarray.arange(2,3);

writeln(a);

writeln(a.sum(0));

writeln(a.sum(1));

writeln(a.mean(0));
writeln(a.mean(1));


var b = ndarray.arange(2,3,4);

writeln(b);

writeln(b.mean(0,1));
writeln(b.mean(1,2));