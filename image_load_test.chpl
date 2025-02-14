use NDArray;

config const imagePath: string;

var a = ndarray.loadImage(imagePath);

writeln(a.shape);

writeln(a);


// var b: ndarray(3,int) = ndarray.loadImage(imagePath,eltType = real(32)):int;
// writeln(b.shape);

// writeln(b);

// a.saveImage("new_sun.png");