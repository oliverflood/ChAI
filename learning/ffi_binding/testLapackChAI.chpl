use Random;
use LapackChAI;
use ChAI;

config const seed = 42;

proc main() {
  var A : [1..3, 1..2] real;
  fillRandom(A, seed);

  var nd = new ndarray(A);

  const (u_nd, s_nd, vt_nd) = svd(nd);
  writeln("U:", u_nd.data);
  writeln("S:", s_nd.data);
  writeln("VT:", vt_nd.data);

  var st: Tensor(real) = Tensor.arange(3, 3);
  const (u_st, s_st, vt_st) = svd(st);
  writeln("U:", u_st.resource.array.data);
  writeln("S:", s_st.resource.array.data);
  writeln("VT:", vt_st.resource.array.data);

  // var dt = new dynamicTensor;
  // const (u_dt, s_dt, vt_dt) = svd(dt);
  // writeln("U:", u_dt.meta.array.data);
  // writeln("S:", s_dt.meta.array.data);
  // writeln("VT:", vt_dt.meta.array.data);
}
