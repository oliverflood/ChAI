// testArgmaxDecode.chpl
use ChAI;      // Likely needed for core types or general setup
use NDArray;   // Where your ndarray type and argmaxDecode method are defined

proc main() {
  // 1. Create a Chapel array with some values
  //    Use real(32) to match typical float precision in LibTorch/ChAI
  var chapArray: [0..2] real(32) = [1.0, 7.0, 3.0]; // Expected argmax index: 1

  // 2. Construct an ndarray from the Chapel array.
  //    This will use your existing `proc init(const Arr: [])` in NDArray.chpl
  var myNdArray = new ndarray(chapArray);

  // 3. Call the argmaxDecode method on the ndarray instance
  //    The `this._tensorHandle` will be created/accessed via `toBridgeTensor`
  //    or the implicit cast `operator :` before calling Bridge.argmaxdecode.
  var argmaxIndex = myNdArray.argmaxDecode();

  // 4. Print the result
  writeln("Chapel array: ", chapArray);
  writeln("Argmax index: ", argmaxIndex); // Should print 1 for the example above

  // Test with another example
  var chapArray2: [0..3] real(32) = [10.0, 5.0, 20.0, 15.0]; // Expected argmax index: 2
  var myNdArray2 = new ndarray(chapArray2);
  var argmaxIndex2 = myNdArray2.argmaxDecode();
  writeln("Chapel array 2: ", chapArray2);
  writeln("Argmax index 2: ", argmaxIndex2); // Should print 2

  // You can keep the accelerator checks for good measure
  if Bridge.acceleratorAvailable() then
    writeln("Accelerator (CUDA/MPS) is available!");
  else
    writeln("Accelerator not available, running on CPU.");

  Bridge.debugCpuOnlyMode(true);
  writeln("Debug CPU only mode is ON. Accelerator available: ", Bridge.acceleratorAvailable());
  Bridge.debugCpuOnlyMode(false);
  writeln("Debug CPU only mode is OFF. Accelerator available: ", Bridge.acceleratorAvailable());

  // IMPORTANT: Ensure tensors are freed!
  // If `myNdArray`'s deinit() correctly calls Bridge.freeBridgeTensorHandle,
  // then memory will be managed. If not, you might need explicit calls or a
  // more robust memory management strategy.
}