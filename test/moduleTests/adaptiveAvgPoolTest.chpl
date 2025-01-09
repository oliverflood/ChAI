use Tensor;
use Network;

config const poolSize = 2;
config const tensorSize  = 5;

// Create a tensor of shape (5, 5, 5)
var X : Tensor(real) = Tensor.arange(tensorSize, tensorSize, tensorSize);
writeln(X);

// Apply the maxPool layer
var model = new AdaptiveAvgPool2D(poolSize);
var output = model(X);

// Write the tensor
writeln(output);
