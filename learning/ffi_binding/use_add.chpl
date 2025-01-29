use CTypes;

extern "add_my_nums" proc add_my_nums(x: c_int, y: c_int): c_int;


proc main() {
    var a: c_int = 5;
    var b: c_int = 15;

    var result = add_my_nums(a, b);
    //var result = a+b;

    writeln("Result of add_my_nums(", a, ", ", b, ") = ", result);
}
