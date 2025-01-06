use Tensor;
use Network;

// Create a tensor of shape (10,)
var X : Tensor(real) = Tensor.arange(10);
writeln(X);

// Apply the linear layer
var model = new Linear(real, 10, 5);
var output = model(X);

// Write the tensor
writeln(output);
