use NDArray;

config const imagePath: string;

var a = ndarray.loadImage(imagePath);

writeln(a.shape);

a.saveImage("new_sun.png");

var randomSun = 0.2 * ndarray.random((...a.shape));


var b = a * (randomSun / 3);
b.saveImage("new_sun2.png");