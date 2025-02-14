use NDArray;

config const imagePath: string;

var a = ndarray.loadImage(imagePath);

writeln(a.shape);

a.saveImage("new_sun.png");

var b = a * (0.2 * ndarray.random((...a.shape)));
b.saveImage("new_sun2.png");