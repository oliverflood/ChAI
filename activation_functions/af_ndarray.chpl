/*
This file contains all activation functions which have not yet been put onto NDArray.chpl --> DynamicTensor.chpl

TODO: ****************
* celu
* leaky_relu
* softshrink
* hardshrink
* softplus

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

inline proc celu(alpha: eltType=1) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = max(0, x) + min(0, alpha * Math.exp(x / alpha) - 1);
    }

    return rl;
}

inline proc leaky_relu(negative_slope: eltType=Math.exp(-2)) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = max(0, x) + negative_slope * min(0, x);
    }

    return rl;
}

inline proc softplus(beta: eltType=1.0, threshold: eltType=20.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltType);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = if x * beta > threshold then x else 1 / beta * log(1 + Math.exp(beta * x));
    }

    return rl;
}

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