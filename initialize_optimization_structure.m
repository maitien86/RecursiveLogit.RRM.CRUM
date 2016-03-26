%   Initialize optimization structure
%%
function [] = initialize_optimization_structure()
    global Op;
    global isLinkSizeInclusive;
    global isFixedUturn;
    global nbobs;
    Op.Optim_Method = OptimizeConstant.TRUST_REGION_METHOD;
    Op.ETA1 = 0.05;
    Op.ETA2 = 0.75;
    Op.maxIter = 500;
    Op.k = 0;
    Op.n = 4;
    if isLinkSizeInclusive == true
        Op.n = Op.n + 1;
    end
    if isFixedUturn == true
        Op.n = Op.n - 1;
    end
    %Op.n = Op.n * 2 + 1 ;
    Op.n = 3 * Op.n;
    Op.x = -ones(Op.n,1) * 1.5;
    Op.x(Op.n) = 0;
    Op.tol = 1e-6;
    Op.radius = 1.0;
    Op.Ak = zeros(Op.n);
    Op.H = eye(Op.n);

end