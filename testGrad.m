global Op;
x = Op.x;
step = 0.0000001;
[val1, gr1] = LL(Op.x);
gr1
h = step * [0 0 1 0 0 0 0 0]';
Op.x = x +h;
[val2, gr2] = LL(Op.x);
(val2 - val1)/step
