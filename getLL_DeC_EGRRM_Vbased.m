%% GRRM LL evaluation
%%
function [LL, grad] = getLL_DeC_EGRRM_Vbased()

    global incidenceFull; 
    global Gradient;
    global Op;
    global Atts;
    global Obs;     % Observation
    global nbobs;  
    global SampleObs;
    global RRMmodel; 

    [lastIndexNetworkState, maxDest] = size(incidenceFull);
    %% Initialize LL  and gradient
    LL = 0;
    grad = zeros(1, Op.n);
   
    %% Compute (S) and deltaS matrix
    dnp = Op.n;
    np = (dnp) / 3 ;
    I = find(incidenceFull);
    [nbnonzero] = size(I,1);
    S = incidenceFull;
    deltaS = objArray(dnp);
    for t = 1: dnp
        deltaS(t).value = S;
    end
    for i = 1:nbnonzero
        [k,a] = ind2sub(size(incidenceFull), I(i));
        col = find(incidenceFull(k,:));
        temp = 0;
        delta = zeros(dnp,1);
        nA = size(col,2);
        nA = 1;
        for j = 1:size(col,2)
            for t = 1: np
                dt = Op.x(2*np + t);
                lambda = dt;
                %u =  Op.x(t) * (Atts(t).Value(k,col(j)) - Atts(t).Value(k,a)) + Op.x(t+np) * Atts(t).Value(k,a);
                u =  Op.x(t) * (Atts(t).Value(k,col(j)) - Atts(t).Value(k,a)) + Op.x(t+np) * Atts(t).Value(k,col(j));
                %u =  -Op.x(t) * Atts(t).Value(k,a);
                %temp = temp+u;
                temp = temp + log(lambda + exp(u));
                delta(t) = delta(t) + (exp(u) * (Atts(t).Value(k,col(j)) - Atts(t).Value(k,a)))/(lambda + exp(u));
                delta(t+np) = delta(t+np) + (exp(u) *  Atts(t).Value(k,col(j)))/(lambda + exp(u));
                delta(t+ 2*np) =  delta(t+ 2*np) + 1/(lambda + exp(u));
            end            
        end
        % Including LC
        S(k,a) = temp/nA;
        for t = 1: dnp
            deltaS(t).value(k,a) = delta(t)/nA;
        end
    end
    %S = -getU(Op.x,false);
    S = sparse(S);
    deltaSR = objArray(Op.n);
    for i = 1:Op.n
        deltaS(i).value = sparse(deltaS(i).value);
        deltaSR(i).value =  deltaS(i).value(1:lastIndexNetworkState,1:lastIndexNetworkState);
        deltaSR(i).value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        deltaSR(i).value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end
    %% Get M, U 
    Ufull = sparse(-S);
    Mfull = sparse(exp(Ufull) .* incidenceFull);
    M = Mfull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    M(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    M(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));  
    % Attribute in regulier size
    AttR = objArray(np);
    for i = 1:np
        AttR(i).value =  Atts(i).Value(1:lastIndexNetworkState,1:lastIndexNetworkState);
        AttR(i).value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        AttR(i).value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end
    %% Compute Z (DeC method)
    N = size(M,1);
    b = sparse(zeros(N,1));
    b(N) = 1;
    B = sparse(zeros(N, maxDest - lastIndexNetworkState));
    B(N,:) = ones(1,maxDest - lastIndexNetworkState);
    for i = 1: maxDest - lastIndexNetworkState
        B(1:lastIndexNetworkState,i) = Mfull(:, i+lastIndexNetworkState);
    end
    [V,boolV] = getV(M,B);
    if (boolV == 0)
            LL = OptimizeConstant.LL_ERROR_VALUE;
            grad = ones(Op.n,1);
            disp('The parameters not fesible')
            return; 
    end
    Z = exp(V);
    %% Compute gradZ 
    gradZ = objArray(Op.n);
    for i = 1:Op.n
        u = -M .* deltaSR(i).value; 
        v = sparse(u * Z); 
        p = -deltaS(i).value(:,lastIndexNetworkState+1 : maxDest) .* Mfull(:,lastIndexNetworkState+1 : maxDest);
        p(lastIndexNetworkState+1,:) = sparse(zeros(1,maxDest - lastIndexNetworkState));        
        p = sparse(p);
        gradZ(i).value =  sparse(A\(v + p)); 
    end
        %% Compute LL value  
    for n = 1:nbobs
        %n
        dest = Obs(n, 1);
        d = dest - lastIndexNetworkState;       
        %% Compute the prob
        path = Obs(n,:);
        lpath = size(find(path),2);
        lnPn = 0;
        Gradient(n,:) =  zeros(1,Op.n);        
        for i = 2:lpath - 1
            maxLink = min(path(i+1),lastIndexNetworkState + 1);
            LnP = (Ufull(path(i),path(i+1)) + log(Z(maxLink,d)) - log(Z(path(i),d)));
            lnPn = lnPn + LnP;
            for j = 1:Op.n
                Gradient(n,j) = Gradient(n,j) - deltaS(j).value(path(i),path(i+1)) +  gradZ(j).value(maxLink,d)/Z(maxLink,d) - gradZ(j).value(path(i),d)/Z(path(i),d);
            end
        end
        LL =  LL + (lnPn - LL)/n;
        grad = grad + (Gradient(n,:) - grad)/n;
        Gradient(n,:) = - Gradient(n,:);
    end   
    LL = -1 * LL; % IN ORDER TO HAVE A MIN PROBLEM
    grad =  - grad';
end

