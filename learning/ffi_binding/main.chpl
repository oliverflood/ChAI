use LAPACK;
use CTypes;
use Random;

const N = 2;
const K = 1;
const seed = 41;


var A : [1..N, 1..N] real;
fillRandom(A, seed);

var X : [1..N, 1..K] real;
fillRandom(X, seed);

var B : [1..N, 1..K] real;

for i in 1..N do
  for j in 1..K do
    for k in 1..N do
      B[i,j] += A[i,k] * X[k,j];

writeln("Matrix A:\n", A, "\n");
writeln("Matrix X:\n", X, "\n");
writeln("Matrix B:\n", B, "\n");

var WorkA = A;
var WorkBX = B;
var ipiv : [1..N] c_int;

var info = gesv(lapack_memory_order.row_major, WorkA, ipiv, WorkBX);

writeln("gesv result for X:\n", WorkBX, "\n");
