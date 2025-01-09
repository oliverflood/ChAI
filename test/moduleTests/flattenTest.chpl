use Tensor;
use Network;

config const tensorSize = 3;

// Create a tensor of shape (3, 3, 3)
var X : Tensor(real) = Tensor.arange(tensorSize, tensorSize, tensorSize);
writeln(X);

// Apply the flatten layer
var model = new Flatten();
var output = model(X);

// Write the tensor
writeln(output);
