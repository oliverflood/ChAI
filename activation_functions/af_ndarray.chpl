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
        rld[i] = if x <= threshold then value else x;
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

inline proc relu6() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = min(max(0, x), 6);
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
        if x > 0 {
            rld[i] = x;
        }

        else {
            rld[i] = alpha * (Math.exp(x) - 1);
        }
    }

    return rl;
}

inline proc selu() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    const alpha: real(64) = 1.6732632423543772848170429916717;
    const scale: real(64) = 1.0507009873554804934193349852946;
    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = scale * (max(0, x) + min(0, alpha * (Math.exp(x) - 1)))
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

// prelu goes here

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

// glu goes here
// gelu goes here

inline proc log_sigmoid() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = log(1 / (1 + Math.exp(-x)));
    }

    return rl;
}

inline proc hardshrink(lambda: real(64)=0.5) {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x if (x > lambda) || (x < lambda) else 0;
    }

    return rl;
}

inline proc tanhshrink() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x - Math.tanh(x);
    }

    return rl;
}

inline proc softsign() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = x / (1 + abs(x));
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

// softmax goes here
// softmin goes here

inline proc softshrink(lambda: real(64)=0.5): throws { // lambda must be non-negative
    if lambda < 0 {
        throw new Error("argument to softshrink function must be non-negative");
    }
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        if x > lambda {
            rld[i] = x - lambda;
        }

        else if x < lambda {
            rld[i] = x + lambda;
        }

        else {
            rld[i] = 0;
        }
    }

    return rl;
}

// gumbel_softmax goes here

inline proc tanh() {
    const ref thisData = data;
    const dom = this.domain;
    var rl = new ndarray(dom, eltype);
    ref rld = rl.data;

    forall i in dom.every() {
        const x = thisData[i];
        rld[i] = Math.tanh(x);
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