use Tensor;


var a = Tensor.zeros(2,3).rrelu();
writeln(a);
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).rrelu();
writeln(b);
Testing.numericPrint(b);

if false {

var c = (Tensor.zeros(10) + 40.0).rrelu();
// writeln(c.degenerateFlatten());
Testing.numericPrint(c);

// same values with lower = -0.001
a = Tensor.zeros(2,3).rrelu(lower=-0.001);
// writeln(a.degenerateFlatten());
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).rrelu(lower=-0.001);
// writeln(b.degenerateFlatten());
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).rrelu(lower=-0.001);
// writeln(c.degenerateFlatten());
Testing.numericPrint(c);

// same values with upper = 10.0
a = Tensor.zeros(2,3).rrelu(upper=10.0);
// writeln(a.degenerateFlatten());
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).rrelu(upper=10.0);
// writeln(b.degenerateFlatten());
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).rrelu(upper=10.0);
// writeln(c.degenerateFlatten());
Testing.numericPrint(c);

// same values with lower = -50.0, upper = 30.0
a = Tensor.zeros(2,3).rrelu(lower = -50.0, upper=30.0);
// writeln(a.degenerateFlatten());
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).rrelu(lower=-50.0, upper=30.0);
// writeln(b.degenerateFlatten());
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).rrelu(lower=-50.0, upper=30.0);
// writeln(c.degenerateFlatten());
Testing.numericPrint(c);
}