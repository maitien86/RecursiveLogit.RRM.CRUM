function [f,g] = LL(x)
    global Op;
    Op.x = x;
    %[f,g] = getLL_DeC_GRRM();
    [f,g] = getLL_DeC_EGRRM();
    %[f,g] = getLL_DeC_EGRRM_Vbased(); 
    %[f g] = getLL_DeC_CRUM();
    Op.nFev  = Op.nFev + 1;
end