use Tensor;

use Network;

import Time;

import Image;

config const detach = true;

config const weightsFolder = "model/";
config const specFile = "model/specification.json";
config const imagePath = "number.png";

var img_ = Image.readImage(imagePath, Image.imageType.png);
var img = [i in img_.domain] img_[i] : real(32);


Tensor.detachMode(detach);

// Construct the model from specification. 
var model: owned Module(real(32)) = loadModel(specFile=specFile,
              weightsFolder=weightsFolder,
              dtype=real(32));
// Print the model's structure. 
writeln(model.signature);

// Load the weights into the model. 
model.loadPyTorchDump(weightsFolder);

var imagend = new ndarray(img);

// writeln(imagend.shape);

var image = new dynamicTensor(imagend.reshape(1,28,28));

// Create array of output results. 

writeln("started");
config const numTimes = 1;
var st = new Time.stopwatch();
st.start();
const pred = model(image).argmax();
st.stop();  

const time = st.elapsed();
writeln("Time: ", time, " seconds.");


writeln("I think it's a ", pred);
