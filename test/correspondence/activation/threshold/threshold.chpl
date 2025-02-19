use Tensor;

var a = Tensor.zeros(2,3).threshold(threshold=-50.0, value=30.0);
Testing.numericPrint(a);

var b = (Tensor.zeros(2,3,4) - 60.0).threshold(threshold=-50.0,value=30.0);
Testing.numericPrint(b);

var c = (Tensor.zeros(10) + 40.0).threshold(threshold=-50.0, value=30.0);
Testing.numericPrint(c);

a = Tensor.zeros(2,3).threshold(threshold=5.0, value=0.0);
Testing.numericPrint(a);

b = (Tensor.zeros(2,3,4) - 60.0).threshold(threshold=-5.0, value=0.0);
Testing.numericPrint(b);

c = (Tensor.zeros(10) + 40.0).threshold(threshold=-5.0, value=0.0);
Testing.numericPrint(c);

a = (Tensor.zeros(10) + 40.0).threshold(threshold=2.0, value=2.0);
Testing.numericPrint(a);