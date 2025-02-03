use Tensor;

var a = Tensor.zeros(2,3).rrelu();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).rrelu();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).rrelu();
writeln(c.degenerateFlatten());

// same values with lower = -0.001
a = Tensor.zeros(2,3).rrelu(lower=-0.001);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).rrelu(lower=-0.001);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).rrelu(lower=-0.001);
writeln(c.degenerateFlatten());

// same values with upper = 10.0
a = Tensor.zeros(2,3).rrelu(upper=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).rrelu(upper=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).rrelu(upper=10.0);
writeln(c.degenerateFlatten());

// same values with lower = -50.0, upper = 30.0
a = Tensor.zeros(2,3).rrelu(lower = -50.0, upper=30.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).rrelu(lower=-50.0, upper=30.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).rrelu(lower=-50.0, upper=30.0);
writeln(c.degenerateFlatten());