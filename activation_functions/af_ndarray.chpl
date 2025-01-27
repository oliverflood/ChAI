/*
This file contains all activation functions which have not yet been put onto NDArray.chpl --> DynamicTensor.chpl

TODO: ****************
* softshrink

Implement the Following:
* prelu
* glu
* softmin
* softmax
* gumbel_softmax
* log_softmax
* ALL NORMS
**********************
*/
// const float_max: eltType = 1.7976931348623157E308;


inline proc softshrink(l: eltType=0.5): throws { // l must be non-negative
    if l < 0 do throw new Error("argument to softshrink function must be non-negative");
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        if x > l then
            rld[i] = x - l;
        else if x < l then
            rld[i] = x + l;
        else
            rld[i] = 0;
    }

    return rl;
}