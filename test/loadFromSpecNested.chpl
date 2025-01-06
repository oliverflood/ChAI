use Tensor;

use Network;

import Time;

config const baseDir = "../";



// Construct the model from specification.
// Currently this doesn't work since the model attributes don't specify that a Conv2D does *not* have a bias, and therefore,
// it assumes that it does have a bias. This is a bug in the model creation from spec code, but difficult to fix, since it
// is missing information about the existence of a bias in the specifications.json file
// See below in "Dummy" where the Conv2D layer is created with bias=false to get around this

// var model: owned Module(real(32)) =  Network.loadModel(specFile=baseDir+"scripts/models/dummy/specification.json",
//             weightsFolder=baseDir+"scripts/models/dummy/",
//             dtype=real(32));

// Construct the model from specification.
// var model2: owned Module(real(32)) =  Network.loadModel(specFile=baseDir+"scripts/models/dummy_two/specification.json",
//             weightsFolder=baseDir+"scripts/models/dummy_two/",
//             dtype=real(32));

type dtype = real(32);

class Dummy : Module(?) {
    var model: owned Sequential(eltType);
    var conv1: owned Conv2D(eltType);

    proc init(type eltType = dtype) {
        super.init(eltType);
        this.model = new Sequential(eltType, (
            new Conv2D(eltType, 3, 64, kernel=7, stride=2, padding=3, bias=false)?,
            new Conv2D(eltType, 64, 64, kernel=3, stride=2, padding=1, bias=true)?
        ), "model");
        this.conv1 = new Conv2D(eltType, 3, 64, kernel=7, stride=2, padding=3, bias=false);
        init this;
        this.moduleName = "dummy";

        for (n,m) in this.moduleFields() {
            addModule(n,m);
        }
    }

    proc forward(input: Tensor(eltType)): Tensor(eltType) {
        return input;
    }
}

class DummyTwo : Module(?) {
    var conv1: owned Conv2D(eltType);
    var newmod: owned Sequential(eltType);

    proc init(type eltType = dtype, ref input_model: owned Dummy(eltType)?) {
        super.init(eltType);
        this.conv1 = new Conv2D(eltType, 3, 64, kernel=7, stride=2, bias=false);
        this.newmod = new Sequential(eltType, (
            input_model,
        ), "newmod");
        init this;
        this.moduleName = "dummy_two";

        for (n, m) in this.moduleFields() {
            addModule(n, m);
        }
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) {
        return this.newmod.forward(input);
    }
}

// Example usage
const dummyModelPath = baseDir+"scripts/models/dummy/"; // The trailing / is important
const dummy2ModelPath = baseDir+"scripts/models/dummy_two/"; // "

var dummy = new Dummy(dtype)?;
dummy!.loadPyTorchDump(dummyModelPath);


var dummyTwo = new DummyTwo(dtype, dummy);
dummyTwo.loadPyTorchDump(dummy2ModelPath);

