%% GRRM LL evaluation
%%
function [LL, grad] = getLL_EGRRM()

    global incidenceFull; 
    global Gradient;
    global Op;
    global Atts;
    global Obs;     % Observation
    global nbobs;  
    global isLinkSizeInclusive;
 
    mu = 1; % MU IS NORMALIZED TO ONE
    [lastIndexNetworkState, maxDest] = size(incidenceFull);
    %% Initialize LL  and gradient
    LL = 0;
    grad = zeros(1, Op.n);
   
    %% Compute (S) and deltaS matrix
    dnp = Op.n;
    np = (dnp-1) / 2;
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
        for j = 1:size(col,2)
            for t = 1: np
                u =  Op.x(t) * Atts(t).Value(k,col(j)) - Op.x(t+np) * Atts(t).Value(k,a);
                %u =  Op.x(t) * Atts(t).Value(k,a) - Op.x(t+np) * Atts(t).Value(k,a);
                temp = temp + log(Op.x(dnp) + exp(u));
                delta(t) = delta(t) + (exp(u) * Atts(t).Value(k,col(j)))/(Op.x(dnp) + exp(u));
                %delta(t) = delta(t) + (exp(u) * Atts(t).Value(k,a))/(Op.x(dnp) + exp(u));                 
                delta(t+np) = delta(t+np) - (exp(u) * Atts(t).Value(k,a))/(Op.x(dnp) + exp(u));
                delta(dnp) =  delta(dnp) + 1/(Op.x(dnp) + exp(u));
            end
        end
        S(k,a) = temp;
        for t = 1: dnp
            deltaS(t).value(k,a) = delta(t);
        end
    end
    S = sparse(S);
    deltaSR = objArray(Op.n);
    for i = 1:Op.n
        deltaS(i).value = sparse(deltaS(i).value);
        deltaSR(i).value =  deltaS(i).value(1:lastIndexNetworkState,1:lastIndexNetworkState);
        deltaSR(i).value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        deltaSR(i).value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end

    %% Compute LL value  
    gradR = objArray(Op.n); 
    gradZ = objArray(Op.n);
    for n = 1:nbobs
        %n
        dest = Obs(n, 1);
        Ufull = sparse(-S);
        Mfull = sparse(exp(Ufull) .* incidenceFull);
        
        M = Mfull(1:lastIndexNetworkState,1:lastIndexNetworkState);            
        addColumn = Mfull(:,dest);
        M(:,lastIndexNetworkState+1) = addColumn;
        M(lastIndexNetworkState+1,:) = zeros(1,lastIndexNetworkState+1);
        M = sparse(M);
        
        U = Ufull(1:lastIndexNetworkState,1:lastIndexNetworkState);            
        addColumn = Ufull(:,dest);
        U(:,lastIndexNetworkState+1) = addColumn;
        U(lastIndexNetworkState+1,:) = zeros(1,lastIndexNetworkState+1);
        U = sparse(U);
         
        for t = 1:Op.n
             addColumn = deltaS(t).value(:,dest);
             deltaSR(t).value(1:lastIndexNetworkState,lastIndexNetworkState+1) = addColumn; 
        end
        % get Z
        [Z, Ok] = getZ(M); % vector with value functions for given beta                                                                     
        if (Ok == 0)
            LL = OptimizeConstant.LL_ERROR_VALUE;
            grad = ones(Op.n,1);
            disp('The parameters not fesible')
            return; 
        end            
        % get gradZ, gradR
        R = -log(Z);
        I = speye(size(M));
        for i = 1:Op.n
            u = - M .* deltaSR(i).value;
            gradZ(i).value = (I-M)\( u * Z);
            gradR(i).value = - gradZ(i).value ./ Z;
        end        
        %% Compute the prob
        path = Obs(n,:);
        lpath = size(find(path),2);
        lnPn = 0;
        Gradient(n,:) =  zeros(1,Op.n);        
        for i = 2:lpath - 1
            maxLink = min(path(i+1),lastIndexNetworkState + 1);
            LnP = Ufull(path(i),path(i+1)) - R(maxLink) + R(path(i));
            lnPn = lnPn + LnP;
            for j = 1:Op.n
                Gradient(n,j) = Gradient(n,j)  - deltaS(j).value(path(i),path(i+1)) - gradR(j).value(maxLink) + gradR(j).value(path(i));
            end
        end
        LL =  LL + (lnPn - LL)/n;
        grad = grad + (Gradient(n,:) - grad)/n;
        Gradient(n,:) = - Gradient(n,:);
    end
    LL = -1 * LL; % IN ORDER TO HAVE A MIN PROBLEM
    grad =  - grad';
end

