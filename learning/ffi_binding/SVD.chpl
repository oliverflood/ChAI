module SVD {
  use LAPACK;

  // High-level SVD function similar to PyTorch's torch.linalg.svd
  proc svd(A: [] real(64), full_matrices: bool = true) {
    const m = A.domain.dim(0).size,
          n = A.domain.dim(1).size,
          min_mn = min(m, n);

    // Copy of A to avoid modifying the original
    var A_copy = A;

    // Always length min_mn
    var S: [1..min_mn] real(64);

    // Create domains for U, VT based on full_matrices
    var domU, domVT: domain(2,int);
    if full_matrices {
      domU  = {1..m, 1..m};
      domVT = {1..n, 1..n};
    } else {
      domU  = {1..m,       1..min_mn};
      domVT = {1..min_mn,  1..n      };
    }

    // Set jobz as reduced ('S') or full ('A') size matrices
    var jobz = if full_matrices then 'A' else 'S';

    // Declare U, VT with the decided shapes
    var U:  [domU]  real(64);
    var VT: [domVT] real(64);

    // Make FFI call to LAPACK_gesdd
    var info: c_int = gesdd(lapack_memory_order.row_major, jobz, A_copy, S, U, VT);

    if info == 0 {
      return (U, S, VT);
    } else if info > 0 {
      halt("SVD did not converge.");
    } else {
      halt("Invalid argument at position ", -info);
    }
  }
}
