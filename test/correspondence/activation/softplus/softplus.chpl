use Tensor;

var a = Tensor.zeros(2,3).softplus();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 60.0).softplus();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 40.0).softplus();
writeln(c.degenerateFlatten());

// same values with beta = -0.001
a = Tensor.zeros(2,3).softplus(beta=-0.001);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 60.0).softplus(beta=-0.001);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).softplus(beta=-0.001);
writeln(c.degenerateFlatten());

// same values with beta = 10.0
a = Tensor.zeros(2,3).softplus(beta=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 60.0).softplus(beta=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 40.0).softplus(beta=10.0);
writeln(c.degenerateFlatten());