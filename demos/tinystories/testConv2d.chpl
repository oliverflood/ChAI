use ChAI;
use NDArray;

proc main() {
  // Construct input tensor: shape (1, 1, 5, 5) for batch=1, channels=1, height=5, width=5
  var inputData: [0..0, 0..0, 0..4, 0..4] real = 
    [[[ [1.0, 2.0, 3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0, 9.0, 10.0],
        [11.0,12.0,13.0,14.0,15.0],
        [16.0,17.0,18.0,19.0,20.0],
        [21.0,22.0,23.0,24.0,25.0] ]]];

  var input = new ndarray(real, inputData.domain, inputData);

  // Kernel tensor: shape (1, 1, 3, 3)
  var kernelData: [0..0, 0..0, 0..2, 0..2] real = 
    [[[ [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0] ]]];

  var kernel = new ndarray(real, kernelData.domain, kernelData);

  // Bias tensor: shape (1)
  var biasData: [0..0] real = [0.0];
  var bias = new ndarray(real, biasData.domain, biasData);

  // Call conv2d with stride=1, padding=1 (same padding)
  var output = input.conv2d(kernel, bias, 1:int(32), 1:int(32));

  writeln("Output shape: ", output.domain);
  writeln("Output tensor:");
  writeln(output);
}

main();
