use Tensor;
use Network;

// Create a tensor of shape (3, 3, 3)
var X : Tensor(real) = Tensor.arange(3, 3, 3);
writeln(X);

// Apply the conv2d layer
var model = new Conv2D(3, 5, 3);
var output = model(X);

// Write the tensor
writeln(output);



