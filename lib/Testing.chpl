module Testing {
    private use NDArray;
    private use StaticTensor;
    private use DynamicTensor;
    private use Utilities.Types;
    private import Utilities as util;
    private import Utilities.Standard;

    private import IO;
    private use IO;
    private use IO.FormattedIO;

    proc numericExport(t: ?tensorType): [] string
            where isSubtype(tensorType, ndarray) 
                    || isSubtype(tensorType, staticTensor)
                    || isSubtype(tensorType, dynamicTensor) {
        type eltType = t.eltType;
        const data = t.degenerateFlatten();
        if eltType == real(32) {
            return [x in data] "%1.32dr".format(x); // "%dr".format(x); // "%.10dr".format(x); // "%dr".format(x); // "%1.14dr".format(x);
        } else {
            util.err("Unsupported type for numericExport for type: " + eltType:string);
        }
    }

    proc numericPrint(t: ?tensorType) 
            where isSubtype(tensorType, ndarray) 
                    || isSubtype(tensorType, staticTensor)
                    || isSubtype(tensorType, dynamicTensor) {
        const data = numericExport(t);
        writeln(data);
    }
}