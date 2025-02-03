use NDArray;

config const imagePath: string;

var a = ndarray.loadImage(imagePath);

writeln(a.shape);

writeln(a);