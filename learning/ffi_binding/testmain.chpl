use Random;
use SVDModule;

config const seed = 41;

proc main() {
  var A : [1..3, 1..2] real;
  fillRandom(A, seed);

  writeln("A's domain: ", A.domain);

  var (U, S, VT) = svd(A, full_matrices = false);

  writeln("Matrix A:\n", A);
  writeln("Matrix U:\n", U);
  writeln("Singular values S:\n", S);
  writeln("Matrix VT:\n", VT);
}
