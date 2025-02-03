module LapackChAI {
  use ChAI;
  use LAPACK;
  use Autograd;
  use NDArray;
  use StaticTensor;
  use DynamicTensor;
  use Utilities.Standard;

  proc raw_svd(A: [] real(64), full_matrices: bool = true) {
    const m      = A.domain.dim(0).size;
    const n      = A.domain.dim(1).size;
    const min_mn = min(m, n);

    var A_copy = A;
    var S: [1..min_mn] real(64);

    var domU, domVT: domain(2, int);
    if full_matrices {
      domU  = {1..m, 1..m};
      domVT = {1..n, 1..n};
    } else {
      domU  = {1..m,       1..min_mn};
      domVT = {1..min_mn,  1..n};
    }
    const jobz = if full_matrices then 'A' else 'S';
    var U:  [domU] real(64);
    var VT: [domVT] real(64);

    const info = gesdd(lapack_memory_order.row_major, jobz, A_copy, S, U, VT);
    if info == 0 then
      return (U, S, VT);
    else if info > 0 then
      halt("SVD did not converge.");
    else
      halt("Invalid argument at position ", -info);
  }

  record svdOp {
    var input: shared BaseTensorResource(?);
    var full_matrices: bool;

    proc init(input: shared BaseTensorResource(?), full_matrices: bool) {
      this.input = input;
      this.full_matrices = full_matrices;
    }

    proc forward() {
      return input.array.svd(full_matrices);
    }

    proc children() { return (input,); }
    proc spec() { return dict(("operation", "SVD")); }
  }

  proc svd(A: ndarray(?), full_matrices: bool = true)
        : (ndarray(?), ndarray(?), ndarray(?)) {
    if A.rank != 2 then
      halt("SVD is only defined for rank=2 NDArray; got rank=", A.rank);

    const castArr = A.data: [A._domain] real(64);
    const (U_arr, S_arr, VT_arr) = raw_svd(castArr, full_matrices);

    var U_nd  = new ndarray(U_arr);
    var S_nd  = new ndarray(S_arr);
    var VT_nd = new ndarray(VT_arr);
    return (U_nd, S_nd, VT_nd);
  }


  proc svd(A: staticTensor(?), full_matrices: bool = true)
        : (staticTensor(?), staticTensor(?), staticTensor(?)) {
    var ctx = new svdOp(A.resource, full_matrices);
    const (u_nd, s_nd, vt_nd) = ctx.forward();

    var u_st = staticTensor(u_nd);
    var s_st = staticTensor(s_nd);
    var vt_st = staticTensor(vt_nd);
    return (u_st, s_st, vt_st);
  }


  proc svd(A: dynamicTensor(?), full_matrices: bool = true)
        : (dynamicTensor(?), dynamicTensor(?), dynamicTensor(?)) {
    for param rank in 1..DynamicTensor.maxRank {
      if A.checkRank(rank) {
        if rank != 2 then
          halt("SVD currently only implemented for rank-2 dynamicTensors; got rank=", rank);
        const st = A.tensorize(rank);

        const (u_st, s_st, vt_st) = svd(st, full_matrices);

        var u_dt = dynamicTensor(u_st);
        var s_dt = dynamicTensor(s_st);
        var vt_dt = dynamicTensor(vt_st);
        return (u_dt, s_dt, vt_dt);
      }
    }
    halt("Could not determine rank in dynamicTensor.svd.");
  }
}
