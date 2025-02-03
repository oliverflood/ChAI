leakuse Tensor;

var a = Tensor.zeros(2,3).leakyrelu();
writeln(a.degenerateFlatten());

var b = (Tensor.zeros(2,3,4) - 1.0).leakyrelu();
writeln(b.degenerateFlatten());

var c = (Tensor.zeros(10) + 4.0).leakyrelu();
writeln(c.degenerateFlatten());

// same values with negative_slope = -0.001
a = Tensor.zeros(2,3).leakyrelu(negative_slope=-0.001);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).leakyrelu(negative_slope=-0.001);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).leakyrelu(negative_slope=-0.001);
writeln(c.degenerateFlatten());

// same values with negative_slope = 10.0
a = Tensor.zeros(2,3).leakyrelu(negative_slope=10.0);
writeln(a.degenerateFlatten());

b = (Tensor.zeros(2,3,4) - 1.0).leakyrelu(negative_slope=10.0);
writeln(b.degenerateFlatten());

c = (Tensor.zeros(10) + 4.0).leakyrelu(negative_slope=10.0);
writeln(c.degenerateFlatten());