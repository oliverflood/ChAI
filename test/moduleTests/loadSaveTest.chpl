use Tensor;
use Network;

// Load the tensor from the file.
var X = Tensor.load("savedTensors/loadInput.chdata");
writeln(X);
X.save("savedTensors/saveOutput.chdata");
