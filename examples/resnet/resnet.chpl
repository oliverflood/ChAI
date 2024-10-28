module Resnet {
    use Tensor;

    use Network;

    config param layerDebug = false;

    type dtype = real;

    // Function to perform batch normalization
    proc batch_norm(X: Tensor(?eltType), gamma: Tensor(eltType), beta: Tensor(eltType),
                    ref moving_mean: Tensor(eltType), ref moving_var: Tensor(eltType),
                    eps: eltType, momentum: eltType): Tensor(eltType) {

        // Reshape gamma, beta, moving_mean, and moving_var to the shape of X
        var shape = (X.tensorize(3))._dom.shape;
        // We want to do two simple calculations but we need to reshape the tensors to the same shape
        // 1. Normalize:
        //     normalized = (X - moving_mean) / sqrt(moving_var + eps)
        // 2. Scale and shift:
        //     output = gamma * normalized + beta

        // 1. Normalize
        // Denominator first
        const epsTensor = Tensor.valueLike(moving_var, eps);
        const movingVarSumEps = moving_var + epsTensor;
        const sqrtMovingVarSumEps = Tensor.sqrt(movingVarSumEps);
        const expandedSqrtMovingVarSumEps = sqrtMovingVarSumEps.reshape(shape[0],1,1).broadcast(shape[0],shape[1],shape[2]);
        // Numerator
        const expandedMovingMean = moving_mean.reshape(shape[0],1,1).broadcast(shape[0],shape[1],shape[2]);
        const numerator = X - expandedMovingMean;
        const normalized = numerator / expandedSqrtMovingVarSumEps;

        // 2.
        // Scale and shift the normalized output using gamma and beta
        const gammaBroadcasted = gamma.reshape(shape[0], 1, 1).broadcast(shape[0],shape[1],shape[2]);
        const betaBroadcasted = beta.reshape(shape[0], 1, 1).broadcast(shape[0],shape[1],shape[2]);
        const output = gammaBroadcasted * normalized + betaBroadcasted;

        /* This is for traning, not inference
            // Compute the mean and variance of the current batch
            // var batchMean = X.mean(axes=(0,2,3), keepDim=true);
            // var batchVar = ((X - batchMean) * (X - batchMean)).mean(axes=(0,), keepDim=true);

            // // Normalize the input using the batch statistics
            // var normalized = (X - batchMean) / Tensor.sqrt(batchVar + epsTensor);


            // Update the running estimates of mean and variance
            // const momemtumTensor = Tensor.valueLike(moving_mean, momentum);
            // const oneMinusMomentum = Tensor.valueLike(moving_mean, 1 - momentum);
            // moving_mean = momemtumTensor * batchMean + (oneMinusMomentum) * moving_mean;
            // moving_var = momemtumTensor * batchVar + (oneMinusMomentum) * moving_var;
        */

        return output;
    }

    class BatchNorm2D : Module(?) {
        var size: int;
        var momentum: eltType;
        var eps: eltType;
        var gamma: owned Parameter(eltType);
        var beta: owned Parameter(eltType);
        var runningMean: owned Parameter(eltType);
        var runningVar: owned Parameter(eltType);

        // Initialize the BatchNorm2D layer
        proc init(type eltType = dtype, size: int, momentum: eltType = 0.1, eps: eltType = 1e-5) {
            super.init(eltType);
            this.size = size;
            this.momentum = momentum;
            this.eps = eps;

            // Initialize gamma to ones and beta to zeros
            this.gamma = new Parameter(Tensor.ones(size) : eltType);
            this.beta = new Parameter(Tensor.zeros(size) : eltType);

            // Initialize running mean and variance
            this.runningMean = new Parameter(Tensor.zeros(size) : eltType);
            this.runningVar = new Parameter(Tensor.ones(size) : eltType);

            init this;
            this.moduleName = "batchnorm";
        }

        override proc setup() {
            addModule("weight",gamma);
            addModule("bias",beta);
            addModule("running_mean",runningMean);
            addModule("running_var",runningVar);

        }


        // Forward pass for the BatchNorm2D layer
        override proc forward(input: Tensor(eltType)): Tensor(eltType) {
            // Call the batch_norm function to perform batch normalization
            return batch_norm(input, gamma.data, beta.data, runningMean.data, runningVar.data, eps, momentum);
        }

        // Return the attributes of the BatchNorm2D layer
        override proc attributes(): moduleAttributes {
            return new moduleAttributes(
                "BatchNorm2D",
                moduleName,
                ("size", size),
                ("momentum", momentum),
                ("eps", eps));
        }
    }


    class BasicBlock: Module(?) {
        const expansion: int;
        var conv1: owned Conv2D(eltType);
        var bn1: owned BatchNorm2D(eltType);
        var conv2: owned Conv2D(eltType);
        var bn2: owned BatchNorm2D(eltType);
        var downsample: owned Sequential(eltType);
        var emptyDownsample: bool;

        proc init(type eltType = dtype, inplans: int, planes: int, stride: int){
            super.init(eltType);
            this.expansion = 1;
            this.conv1 = new Conv2D(eltType, inplans, planes, kernel=3, stride=stride, bias=false);
            this.bn1 = new BatchNorm2D(eltType, size=planes);
            this.conv2 = new Conv2D(eltType, planes, planes, kernel=3, stride=1, bias=false);
            this.bn2 = new BatchNorm2D(eltType, size=planes);
            if stride != 1 || inplans != planes {
                this.downsample = new Sequential(eltType,(
                    new Conv2D(eltType, inplans, planes, kernel=1, stride=stride, bias=false)?,
                    new BatchNorm2D(eltType, size=planes)?)
                    , "downsample"
                    );
            } else {
                this.downsample = new Sequential(eltType, true, "downsample");
                this.emptyDownsample = true;
            }

            init this;
            this.moduleName = "basicblock";

            for (n,m) in moduleFields() {
                addModule(n,m);
            }
        }

        override proc forward(input: Tensor(eltType)): Tensor(eltType) {
            var identity = input;
            var outVal = this.conv1(input);
            outVal = this.bn1(outVal);
            outVal = outVal.relu();
            outVal = this.conv2(outVal);
            outVal = this.bn2(outVal);
            if !this.emptyDownsample {
                identity = this.downsample(identity);
            }
            outVal = outVal + identity;
            outVal = outVal.relu();
            return outVal;
        }
    }



    class Bottleneck: Module(?) {
        const expansion: int;
        var conv1: owned Conv2D(eltType);
        var bn1: owned BatchNorm2D(eltType);
        var conv2: owned Conv2D(eltType);
        var bn2: owned BatchNorm2D(eltType);
        var conv3: owned Conv2D(eltType);
        var bn3: owned BatchNorm2D(eltType);
        var downsample: owned Sequential(eltType);
        var emptyDownsample: bool;


        proc init(type eltType = dtype, inplanes:int, planes:int, stride:int) {
            super.init(eltType);
            this.expansion = 4;
            var width = planes; // width = int(planes * (base_width / 64.0)) * groups but groups = 1 and base_width = 64
            this.conv1 = new Conv2D(eltType, inplanes, width, kernel=1, stride=1, bias=false);
            this.bn1 = new BatchNorm2D(eltType, size=width);
            this.conv2 = new Conv2D(eltType, width, width, kernel=3, stride=stride, padding=1, bias=false); // we don't have padding implemented yet
            this.bn2 = new BatchNorm2D(eltType, size=width);
            this.conv3 = new Conv2D(eltType, width, planes * expansion, kernel=1, bias=false);
            this.bn3 = new BatchNorm2D(eltType, size=planes * expansion);
            if stride !=1 || inplanes != planes * expansion {
                this.downsample = new Sequential(eltType,(
                    new Conv2D(eltType, inplanes, planes * expansion, kernel=1, stride=stride, bias=false)?,
                    new BatchNorm2D(eltType, size=planes * expansion)?)
                    , "downsample"
                    );
            } else {
                this.downsample = new Sequential(eltType, true, "downsample");
                this.emptyDownsample = true;
            }

            init this;
            this.moduleName = "bottleneck";

            for (n,m) in moduleFields() {
                if !(emptyDownsample && n == "downsample") {
                    addModule(n,m);
                }
            }
        }

        override proc forward(input: Tensor(eltType)): Tensor(eltType) {
            var identity = input;
            var outVal = this.conv1(input);
            outVal = this.bn1(outVal);
            outVal = outVal.relu();

            outVal = this.conv2(outVal);
            outVal = this.bn2(outVal);
            outVal = outVal.relu();

            outVal = this.conv3(outVal);
            outVal = this.bn3(outVal);

            if !this.emptyDownsample {
                identity = this.downsample(identity);
            }

            outVal = outVal + identity;
            outVal = outVal.relu();
            return outVal;
        }
    }


    class ResNet50: Module(?) {
        var inplanes: int;
        var groups: int;
        var baseWidth: int;
        var conv1: owned Conv2D(eltType);
        var bn1: owned BatchNorm2D(eltType);
        var maxpool: owned MaxPool(eltType);
        var layer1: owned Sequential(eltType);
        var layer2: owned Sequential(eltType);
        var layer3: owned Sequential(eltType);
        var layer4: owned Sequential(eltType);
        var avgpool: owned AdaptiveAvgPool2D(eltType);
        var fc: owned Linear(eltType);

        proc init(type eltType, numClasses, groups=1, widthPerGroup=64){
            super.init(eltType);
            this.inplanes = 64;
            this.groups = groups;
            this.baseWidth = widthPerGroup;
            this.conv1 = new Conv2D(eltType, 3, this.inplanes, kernel=7, stride=2, padding=3, bias=false);
            this.bn1 = new BatchNorm2D(eltType, size=this.inplanes);
            this.maxpool = new MaxPool(eltType, poolSize=3, stride=2, padding=1, dilation=1);
            this.layer1 = new Sequential(eltType,(
                new Bottleneck(eltType, 64, 64, 1)?,
                new Bottleneck(eltType, 256, 64, 1)?,
                new Bottleneck(eltType, 256, 64, 1)?),
                "layer1"
                );
            this.layer2 = new Sequential(eltType,(
                new Bottleneck(eltType, 256, 128, 2)?,
                new Bottleneck(eltType, 512, 128, 1)?,
                new Bottleneck(eltType, 512, 128, 1)?,
                new Bottleneck(eltType, 512, 128, 1)?),
                "layer2"
                );
            this.layer3 = new Sequential(eltType,(
                new Bottleneck(eltType, 512, 256, 2)?,
                new Bottleneck(eltType, 1024, 256, 1)?,
                new Bottleneck(eltType, 1024, 256, 1)?,
                new Bottleneck(eltType, 1024, 256, 1)?,
                new Bottleneck(eltType, 1024, 256, 1)?,
                new Bottleneck(eltType, 1024, 256, 1)?),
                "layer3"
                );
            this.layer4 = new Sequential(eltType,(
                new Bottleneck(eltType, 1024, 512, 2)?,
                new Bottleneck(eltType, 2048, 512, 1)?,
                new Bottleneck(eltType, 2048, 512, 1)?),
                "layer4"
                );
            this.avgpool = new AdaptiveAvgPool2D(eltType, outputSize=1);
            this.fc = new Linear(eltType, 2048, numClasses);

            init this;
            this.moduleName = "resnet50";

            for (n,m) in this.moduleFields() {
                addModule(n,m);
            }
        }

        override proc forward(input: Tensor(eltType)): Tensor(eltType) {
            var x = input;
            x = this.conv1(input);
            x = this.bn1(x);
            x = x.relu();
            x = x.maxPool(3);
            x = this.layer1(x);
            x = this.layer2(x);
            x = this.layer3(x);
            x = this.layer4(x);
            x = this.avgpool(x);
            x = x.flatten();
            x = this.fc(x);
            return x;
        }

    }

}  // End of module Resnet


// Use Resnet50

// var model = Network.loadModel(specFile="./models/resnet50/specification.json",
//               weightsFolder="./models/resnet50/",
//               dtype=dtype);

module LoadTest {
    use Resnet;

    // proc main(){
    //     const modelPath = "models/resnet50/";

    //     var resnet = new ResNet50(dtype, numClasses=1000);


    //     for (n,_) in resnet.moduleFields() {
    //         writeln(n);
    //         // writeln("\t", m);
    //     }

    //     resnet.loadPyTorchDump(modelPath);
    // }
}
