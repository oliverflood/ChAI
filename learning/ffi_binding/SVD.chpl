module SVD {
    use LAPACK;
    use CTypes;

    // High-level SVD function similar to PyTorch's torch.linalg.svd
    proc svd(A: [] real(64), compute_uv: bool = true, full_matrices: bool = true) {
        const m = A.domain.dim(0).size,
              n = A.domain.dim(1).size,
              min_mn = min(m, n);

        // Copy of A to avoid modifying the original
        var A_copy = A;

        // Arrays for singular values, left singular vectors (U), and right singular vectors (VT)
        var S: [1..min_mn] real(64);
        var U: [1..m, 1..m] real(64);
        var VT: [1..n, 1..n] real(64);

        // Workspace and info variable
        var info: c_int;

        // Call LAPACK's gesdd function for SVD
        info = gesdd(lapack_memory_order.row_major, 'A', A_copy, S, U, VT);

        if info == 0 {
            return (U, S, VT);
        } else if info > 0 {
            halt("SVD did not converge.");
        } else {
            halt("Invalid argument at position ", -info);
        }
    }
}
