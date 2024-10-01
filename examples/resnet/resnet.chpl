use Tensor;

use Network;

config param layerDebug = false;

type dtype = real(32);


// Dummy Batch norm that doesn't actully do the real math
// This is just a placeholder for the real implementation
class BatchNorm : Module(?) {
    proc init(type eltType = real, size:int) {
        super.init(eltType);
        init this;
        this.moduleName = "batchnorm";
    }

    override proc forward(input: Tensor(eltType)): Tensor(eltType) do
        return input;

    override proc attributes(): moduleAttributes {
        return new moduleAttributes(
            "BatchNorm",
            moduleName,
            ("size", size));
    }
}


class BasicBlock: Module(?) {
    const expansion: int;
    var conv1: owned Conv2D(eltType);
    var bn1: owned BatchNorm(eltType);
    var conv2: owned Conv2D(eltType);
    var bn2: owned BatchNorm(eltType);
    var downsample: owned Sequential(eltType);

    proc init(type eltType = dtype, inplans: int, planes: int, stride: int){
        super.init(eltType);
        this.expansion = 1;
        this.conv1 = new Conv2D(eltType, inplans, planes, kernel=3, stride=stride, bias=false);
        this.bn1 = new BatchNorm(eltType, size=planes);
        this.conv2 = new Conv2D(eltType, planes, planes, kernel=3, stride=1, bias=false);
        this.bn2 = new BatchNorm(eltType, size=planes);
        if stride != 1 || inplans != planes {
            this.downsample = new Sequential(eltType,(
                new Conv2D(eltType, inplans, planes, kernel=1, stride=stride, bias=false)?,
                new BatchNorm(eltType, size=planes)?)
                );
        } else {
            this.downsample = new Sequential(eltType);
        }

        init this;
        this.moduleName = "basicblock";

        for (n,m) in moduleFields() {
            addModule(n,m);
        }
    }

    proc forward(input: Tensor(eltType)): Tensor(eltType) {
        var identity = input;
        var outVal = this.conv1(input);
        outVal = this.bn1(outVal);
        outVal = outVal.relu();
        outVal = this.conv2(outVal);
        outVal = this.bn2(outVal);
        if this.downsample.moduleList.len() != 0 {
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
    var bn1: owned BatchNorm(eltType);
    var conv2: owned Conv2D(eltType);
    var bn2: owned BatchNorm(eltType);
    var conv3: owned Conv2D(eltType);
    var bn3: owned BatchNorm(eltType);
    var downsample: owned Sequential(eltType);


    proc init(type eltType = dtype, inplanes:int, planes:int, stride:int) {
        super.init(eltType);
        this.expansion = 4;
        var width = planes; // width = int(planes * (base_width / 64.0)) * groups but groups = 1 and base_width = 64
        this.conv1 = new Conv2D(eltType, inplanes, width, kernel=1, stride=1, bias=false);// we don't have bias arg implemented yet
        this.bn1 = new BatchNorm(eltType, size=width);
        this.conv2 = new Conv2D(eltType, width, width, kernel=3, stride=stride/*, padding=1*/, bias=false); // we don't have padding implemented yet
        this.bn2 = new BatchNorm(eltType, size=width);
        this.conv3 = new Conv2D(eltType, width, planes * expansion, kernel=1, bias=false);
        this.bn3 = new BatchNorm(eltType, size=planes * expansion);
        if stride !=1 || inplanes != planes * expansion {
            this.downsample = new Sequential(eltType,(
                new Conv2D(eltType, inplanes, planes * expansion, kernel=1, stride=stride, bias=false)?,
                new BatchNorm(eltType, size=planes * expansion)?)
                );
        } else {
            this.downsample = new Sequential(eltType);
        }

        init this;
        this.moduleName = "bottleneck";

        for (n,m) in moduleFields() {
            addModule(n,m);
        }
    }

    proc forward(input: Tensor(eltType)): Tensor(eltType) {
        var identity = input;
        var outVal = this.conv1(input);
        outVal = this.bn1(outVal);
        outVal = outVal.relu();
        outVal = this.conv2(outVal);
        outVal = this.bn2(outVal);
        outVal = outVal.relu();
        outVal = this.conv3(outVal);
        outVal = this.bn3(outVal);
        if this.downsample.moduleList.len() != 0 {
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
    var bn1: owned BatchNorm(eltType);
    var layer1: owned Sequential(eltType);
    var layer2: owned Sequential(eltType);
    var layer3: owned Sequential(eltType);
    var layer4: owned Sequential(eltType);
    var fc: owned Linear(eltType);

    proc init(type eltType, numClasses, groups=1, widthPerGroup=64){
        super.init(eltType);
        this.inplanes = 64;
        this.groups = groups;
        this.baseWidth = widthPerGroup;
        this.conv1 = new Conv2D(eltType, 3, 64, kernel=7, stride=2, bias=false);
        this.bn1 = new BatchNorm(eltType, size=64);
        this.layer1 = new Sequential(eltType,( // TODO Rename all the sequential modules so they pick up the right names in files
            new Bottleneck(eltType, 64, 64, 1)?,
            new Bottleneck(eltType, 256, 64, 1)?,
            new Bottleneck(eltType, 256, 64, 1)?)
            );
        this.layer2 = new Sequential(eltType,(
            new Bottleneck(eltType, 256, 128, 2)?,
            new Bottleneck(eltType, 512, 128, 1)?,
            new Bottleneck(eltType, 512, 128, 1)?,
            new Bottleneck(eltType, 512, 128, 1)?)
            );
        this.layer3 = new Sequential(eltType,(
            new Bottleneck(eltType, 512, 256, 2)?,
            new Bottleneck(eltType, 1024, 256, 1)?,
            new Bottleneck(eltType, 1024, 256, 1)?,
            new Bottleneck(eltType, 1024, 256, 1)?,
            new Bottleneck(eltType, 1024, 256, 1)?,
            new Bottleneck(eltType, 1024, 256, 1)?)
            );
        this.layer4 = new Sequential(eltType,(
            new Bottleneck(eltType, 1024, 512, 2)?,
            new Bottleneck(eltType, 2048, 512, 1)?,
            new Bottleneck(eltType, 2048, 512, 1)?)
            );
        this.fc = new Linear(eltType, 2048, numClasses);

        init this;
        this.moduleName = "resnet50";

        for (n,m) in this.moduleFields() {
            addModule(n,m);
        }
    }

    proc forward(input: Tensor(eltType)): Tensor(eltType) {
        var x = this.conv1(input);
        x = this.bn1(x);
        x = x.relu();
        x = x.maxPool(3);
        x = this.layer1(x);
        x = this.layer2(x);
        x = this.layer3(x);
        x = this.layer4(x);
        // Missing mean and std
        // # out = out.mean([2,3])
        x = this.fc(x);
        return x;
    }

}


// Use Resnet50

// var model = Network.loadModel(specFile="./models/resnet50/specification.json",
//               weightsFolder="./models/resnet50/",
//               dtype=dtype);


const modelPath = "models/resnet50/";

var resnet = new ResNet50(dtype, numClasses=1000);


for (n,_) in resnet.moduleFields() {
    writeln(n);
    // writeln("\t", m);
}

resnet.loadPyTorchDump(modelPath);