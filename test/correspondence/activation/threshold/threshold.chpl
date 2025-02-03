use Tensor;

var a = Tensor.zeros(2,3).threshold();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).threshold();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).threshold();
writeln(c.degenerateFlatten());

// same values with threshold = -10.0
a = Tensor.zeros(2,3).threshold(threshold=-10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).threshold(threshold=-10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).threshold(threshold=-10.0);
writeln(c.degenerateFlatten());

// same values with value = 10.0
a = Tensor.zeros(2,3).threshold(value=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).threshold(value=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).threshold(value=10.0);
writeln(c.degenerateFlatten());

// same values with threshold = -50.0, value = 30.0
a = Tensor.zeros(2,3).threshold(threshold = -50.0, value=30.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).threshold(threshold=-50.0, value=30.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).threshold(threshold=-50.0, value=30.0);
writeln(c.degenerateFlatten());