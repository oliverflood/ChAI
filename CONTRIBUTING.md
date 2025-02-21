# Contributing


Suppose you want to raise the elements of a tensor $T\in\mathbb{T}^n(\mathbb{F})$ by a power $a\in \mathbb{F}$ and multiply the result by a scalar $b \in \mathbb{F}$, via the map
$$
t_i \mapsto b \cdot {t_i}^a
$$
where $t_i = T(i) \in \mathbb{F}$ for $i \in \textsf{dom}(T)$. We will call this operation a power operation of exponent $a$ and scalar $b$, and in Chapel notation, it will be written as `.pow(a, b)`.

You have two options to implement this in ChAI: the easy way (less performant) and the preferred way (more performant), which is recommended.

In either case, you will be implementing at least three instance methods for `ndarray`, `staticTensor`, and `dynamicTensor`:
- `lib/NDarray.chpl`
```chapel
proc ndarray.pow(a: eltType, b: eltType): ndarray(rank,eltType) { ... }
```
- `lib/StaticTensor.chpl`
```chapel
proc staticTensor.pow(a: eltType, b: eltType): staticTensor(rank,eltType) { ... }
```
- `lib/DynamicTensor.chpl`
```chapel
proc dynamicTensor.pow(a: eltType, b: eltType): dynamicTensor(eltType) { ... }
```


## The *preferred* way

The prefered way to implement the power operation is to write the numerically efficient element-wise kernel completely in `lib/NDArray.chpl` as the function 
```chapel
proc ndarray.pow(a: eltType, b: eltType): ndarray(rank,eltType) {
    const dom = this.domain;          // dom(T)
    var u = new ndarray(dom,eltType);
    const ref tData = this.data;      // T
    ref uData = u.data;               // U
    forall i in dom.every() {
        const ref ti = tData[i];      // t_i
        aData[i] = b * (ti ** a);     // u_i = b * t_i^a
    }
    return u;
}
```
The `**` operator is the exponentiation operator in Chapel. The `forall` loop is a parallel loop that will be executed in parallel on the CPU (or GPU, if available). Each iteration $i \in \textsf{dom}(T) = \textsf{dom}(U)$ will be computed in parallel and correspond to the computation of a single element $u_i$ of the output tensor $U$. 