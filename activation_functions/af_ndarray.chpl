/*
TODO: ****************
* prelu
* glu
* softmin
* softmax
* gumbel_softmax
* log_softmax
* ALL NORMS
**********************
*/
inline proc threshold(threshold: real(64), value: real(64)) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x if x > threshold else value;
    }

    return rl;
}

inline proc hardtanh(min_val: real(64)=-1.0, max_val: real(64)=1.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        if x > max_val {
            rld[i] = max_val;
        }

        else if x < min_val {
            rld[i] = min_val;
        }

        // otherwise no change in the value
        else {
            rld[i] = x;
        }
    }

    return rl;
}

inline proc hardswish() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        if x <= -3 {
            rld[i] = 0;
        }

        else if x >= 3 {
            rld[i] = x;
        }

        else {
            rld[i] = (x * (x + 3)) / 6
        }
    }

    return rl;
}

inline proc elu(alpha: real(64)=1) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x if x > 0 else alpha * (Math.exp(x) - 1);
    }

    return rl;
}

inline proc celu(alpha: real(64)=1) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = max(0, x) + min(0, alpha * Math.exp(x / alpha) - 1);
    }

    return rl;
}

inline proc leaky_relu(negative_slope: real(64)=Math.exp(-2)) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = max(0, x) + negative_slope * min(0, x);
    }

    return rl;
}

inline proc rrelu(lower: real(64)=0.125, upper: real(64)=1.0/3.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    var a: [dom] real(64);
    fillRandom(a);
    forall i in dom.every() {
        const x = thisData[i];
        // var a: [0..0] real(64); // singular random value
        // fillRandom(a);
        a = 0.125 + (1.0 / 3.0 - 0.125) * a; // scale it so that it is between 1/8, 1/3
        rld[i] = max(0, x) + min(0, a * x);
    }

    return rl;
}

inline proc hardshrink(l: real(64)=0.5) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x if (x > l) || (x < l) else 0;
    }

    return rl;
}

inline proc softplus(beta: real(64)=1.0, threshold: real(64)=20.0) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        // if x * beta > threshold, revert to linear
        if x * beta > threshold {
            rld[i] = x;
        }

        else {
            rld[i] = 1 / beta * log(1 + Math.exp(beta * x));
        }
    }

    return rl;
}

inline proc softshrink(l: real(64)=0.5): throws { // l must be non-negative
    if l < 0 {
        throw new Error("argument to softshrink function must be non-negative");
    }
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        if x > l {
            rld[i] = x - l;
        }

        else if x < l {
            rld[i] = x + l;
        }

        else {
            rld[i] = 0;
        }
    }

    return rl;
}

inline proc hard_sigmoid() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        if x <= -3 {
            rld[i] = 0;
        }

        else if x >= 3 {
            rld[i] = 1;
        }

        else {
            rld[i] = x/6.0 + 0.5;
        }
    }

    return rl;
}