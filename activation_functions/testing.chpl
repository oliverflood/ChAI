use IO;
use Random;
use Math;

proc main() {
    writeln("in testing function");

    var D: domain(2) = {2..7, 1..4};
    // writeln("specific dimension: ", D.dim(0));
    // writeln("low0: ", D.dim(0).low);
    // writeln("high0: ", D.dim(0).high);
    // writeln("size0: ", D.dim(0).size);
    // writeln("\n");
    // writeln("low1: ", D.dim(1).low);
    // writeln("high1: ", D.dim(1).high);
    // writeln("size1: ", D.dim(1).size);
    var d0 = D.dim(0);
    var d1 = D.dim(1);
    writeln("low0: ", d0.low);
    writeln("high0: ", d0.high);
    writeln("size0: ", d0.size);
    writeln("\n");
    writeln("low1: ", d1.low);
    writeln("high1: ", d1.high);
    writeln("size1: ", d1.size);
}