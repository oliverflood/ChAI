# Contributing


#### Dimension
A positive integer $n$ is a *dimension*. The set of all dimensions is denoted by $\mathbb{N}^+ = \mathbb{N} \setminus \{0\}$.

#### Shape
For a dimension $n \in \mathbb{N}^+$, a tuple $s \in \mathbb{N}^n$ is a *shape of dimension $n$*.

#### Domain
For a shape $s \in \mathbb{N}^n$, the *domain* of $s$ is the set of all tuples, 
<!-- $i \in \mathbb{N}^n$ such that $i_k < s_k$ for $k \in \{1,2,\ldots,n\}$. -->

$$
\textsf{dom}(s) = \{i \in \mathbb{N}^n \mid \forall k\in \mathbb{N}, 1 \leq k \leq n \implies k \in i_k < s_k\}.
$$

<!-- $$
\textsf{dom}(s) = \{i \in \mathbb{N}^n \mid i_k < s_k \text{ for } k \in \{1,2,\ldots,n\}\}.
$$ -->

For a shape $s \in \mathbb{N}^n$, the *domain* of $s$ is the set of all tuples $i \in \mathbb{N}^n$ such that $i_k < s_k$ for $k \in \{1,2,\ldots,n\}$.

A tuple $s \in \mathbb{N}^n$ is a *shape* if $n$ is a dimension.




# h


# HELLO


 The set of all shapes is denoted by $\mathbb{N}^+_\text{shape} = \mathbb{N}^+ \times \mathbb{N}^+ \times \cdots \times \mathbb{N}^+$.
A *shape* is a tuple of dimensions, that is, .
A *shape* is a tuple of dimensions, that is, the set of all shapes is denoted by $\mathbb{N}^+_\text{shape} = \mathbb{N}^+ \times \mathbb{N}^+ \times \cdots \times \mathbb{N}^+$.

Suppose you want to raise the elements of a tensor $T\in\mathbb{T}^n(\mathbb{F})$ by a power $a\in \mathbb{F}$ and multiply the result by a scalar $b \in \mathbb{F}$, via the map:
$$
t_i \mapsto b \cdot {t_i}^a
$$
where $t_i = T(i) \in \mathbb{F}$ for $i \in \textsf{dom}(T)$. 
We will call this operation a power operation of exponent $a$ and scalar $b$, and write 
$$
\text{pow}(T, a, b) = \left[b \cdot {t_i}^a\right]_{i \in \textsf{dom}(T)} = \left[u_i\right]_{i \in \textsf{dom}(U)} = U.
$$
where $U\in\mathbb{T}^n(\mathbb{F})$ is the output tensor, where $u_i = b \cdot {t_i}^a=\text{pow}(t_i,a,b)$, since $\textsf{dom}(T) = \textsf{dom}(U)$. In Chapel notation, we will write `.pow(a, b)`.

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


Next, you will need to implement the `pow` method for `staticTensor` and `dynamicTensor` in `lib/StaticTensor.chpl` and `lib/DynamicTensor.chpl`, respectively. To do this, you need to create a representation for `pow` within the autograd system. This is done by creating a new `powOp` record/struct in `lib/Autograd.chpl`:
```chapel
record powOp : serializable {
    var input: shared BaseTensorResource(?);
    var a: input.eltType;
    var b: input.eltType;

    proc children do return (input,);

    proc forward() do
        return input.array.pow(a, b);

    proc backward(grad: ndarray(?rank,?eltType)): ndarray(rank,eltType) 
            where rank == input.rank
                && eltType == input.eltType {
        const ref dLdU = grad.data;
        const ref U = input.array.data;
        const dom = input.array.domain;
        var dLdT = new ndarray(dom, eltType);
        // const aMinusOne = a - 1;
        // const abProd = a * b;
        forall i in dom.every() {
            const ref ui = U[i];
            const ref dldui = dLdU[i];
            // const duidti = abProd * (ui ** aMinusOne);
            const duidti = (a * b) * (ui ** (a - 1));
            dLdT[i] = dldui * duidti;
        }
        return dLdT;
    }

    proc spec : GradOpSpec do 
        return new dict(
            ("operation","Pow"),
            ("a",a),
            ("b",b)
        );
}
```
The `powOp` struct is a record that contains the input tensor, the exponent $a$, and the scalar $b$. The `forward` method computes the forward pass of the power operation, while the `backward` method computes the backward pass. The `spec` method returns a dictionary that contains the operation name and the values of $a$ and $b$.

The backward pass is computed via
$$
\frac{\partial L}{\partial T} = \frac{\partial L}{\partial U} \cdot \frac{\partial U}{\partial T}
$$
where $L$ is the loss, $U$ is the output tensor, and $T$ is the input tensor. The tensor $\frac{\partial L}{\partial U}$ is the gradient of the loss with respect to the output tensor, and $\frac{\partial U}{\partial T}$ is the derivative of the output tensor with respect to the input tensor. $\frac{\partial L}{\partial U}$ is given as input `grad` to `powOp.backward`, so then we are to compute 
$$
\frac{\partial L}{\partial T} = \frac{\partial L}{\partial U} \cdot \frac{\partial U}{\partial T},
$$
so the hard part is to find $\frac{\partial U}{\partial T}$.

Since $L$ is a scalar, $\frac{\partial L}{\partial U}$ is a tensor of the same shape as $U$, that is $$\frac{\partial L}{\partial U} = \left[\frac{\partial L}{\partial u_i}\right]_{i \in \textsf{dom}(U)},$$ so $$\textsf{dom}(U) = \textsf{dom}\left(\frac{\partial L}{\partial U}\right).$$
Then since each element $u_i$ in $U$ is dependent on the corresponding $t_i$ in $T$, and since $\textsf{dom}(T) = \textsf{dom}(U)$, the derivative $\frac{\partial U}{\partial T}$ is the same shape as $T$, $U$, and $\frac{\partial L}{\partial U}$. That is,
$$
\textsf{dom}(T) = \textsf{dom}(U) = \textsf{dom}\left(\frac{\partial U}{\partial T}\right) = \textsf{dom}\left(\frac{\partial L}{\partial U}\right).
$$
Therefore, we can write
$$
\frac{\partial L}{\partial U} = \left[\frac{\partial L}{\partial u_i}\right]_{i \in \textsf{dom}(U)}
\qquad \text{and} \qquad
\frac{\partial U}{\partial T} = \left[\frac{\partial u_i}{\partial t_i}\right]_{i \in \textsf{dom}(T)}.
$$

The derivative $\frac{\partial U}{\partial T}$ is computed as
$$
\frac{\partial U}{\partial T} 
= \left[\frac{\partial u_i}{\partial t_i}\right]_{i \in \textsf{dom}(T)} 
= \left[\frac{\partial}{\partial t_i}\left(b\cdot {t_i}^a\right)\right]_{i \in \textsf{dom}(T)} 
= \left[b\cdot\frac{\partial}{\partial t_i}{t_i}^a\right]_{i \in \textsf{dom}(T)} 
= \left[b\cdot a \cdot {t_i}^{a - 1}\right]_{i \in \textsf{dom}(T)} 
= \left[b a {t^{a-1}}\right]_{i \in \textsf{dom}(T)}
$$
so we have
$$
\frac{\partial U}{\partial T} = b a {t^{a-1}}.
$$

Finally, you need to add the `pow` method to the `staticTensor` and `dynamicTensor` records:
- `lib/StaticTensor.chpl`
```chapel
proc staticTensor.pow(a: eltType, b: eltType): staticTensor(rank,eltType) {
    const ctx = new powOp(meta,a,b);
    return new tensorFromCtx(rank,eltType,ctx);
}
```
- and `lib/DynamicTensor.chpl`
```chapel
proc dynamicTensor.hardTanh(a: eltType, b: eltType): dynamicTensor(eltType) {
    for param rank in 1..maxRank {
        if this.checkRank(rank) then
            return this.forceRank(rank).pow(a,b).eraseRank();
    }
    halt("Could not determine rank in dynamicTensor.pow.");
    return new dynamicTensor(eltType);
}
```