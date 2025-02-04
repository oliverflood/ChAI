use NDArray;
import Random;

var a = ndarray.random(shape=(5,5),seed=1);

writeln(a);

// var b = ndarray.random(5,5);
// writeln(b);

var rs = new Random.randomStream(real(32),1);
var c = new ndarray(real(32),rs,{0..<5});
writeln(c);
writeln(new ndarray(real(32),rs,{0..<5}));