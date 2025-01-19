// use NDArray;
use IO;
use Random;
use Math;

proc temp(num: real(64)=10.0) {
    return num;
}

proc greeting(name: string="World") {
    writeln("Hello ", name);
}

proc main() {
    writeln("in testing function");
    // greeting();
    // greeting("Jose");
    // writeln(temp());
    // writeln(temp(-20.0));

    // var tvar: real(64) = 1.0/3;
    // writeln(tvar);

    // var bad_count: int = 0;
    // var low: real(64) = 1.0;
    // var high: real(64) = 0.0;
    // for i in 1..1000 {
    //     var a: [0..0] real(64); // singular random value
    //     fillRandom(a);
    //     a = 0.125 + (1.0/3.0 - 0.125) * a;
    //     if (a[0] < 0.125) || (a[0] > 1.0 / 3.0) {
    //         bad_count += 1;
    //     }

    //     if a[0] < low {
    //         low = a[0];
    //     }

    //     if a[0] > high {
    //         high = a[0];
    //     }

    // }
    // writeln("bad_count: ", bad_count, " lowest: ", low, " highest: ", high);

    // writeln(Math.exp(1));
    // writeln(max(0, 3, 7));

    var num: int = -1;
    var x: int = 0;
    // num = 0 if x == 0 else 1;
    num = if x == 0 then 0 else 1;
    writeln("num: ", num, "\tx: ", x);

    x = 1;
    num = if x == 0 then 0 else 1;
    writeln("num: ", num, "\tx: ", x);
}