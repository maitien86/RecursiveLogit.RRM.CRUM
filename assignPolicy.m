% RRM-RUM based combined method recursive logit model.
% Compute the loglikelohood value and its gradient.
%%
function [] = assignPolicy()

    global incidenceFull; 
    global Gradient;
    global Op;
    global Atts;
    global Obs;     % Observation
    global nbobs;  
    global isLinkSizeInclusive;
    global Policy;
    
    %% ----------------------------------------------------
    mu = 1; % MU IS NORMALIZED TO ONE
    [lastIndexNetworkState, nsize] = size(incidenceFull);
   
    %% Compute (S) and deltaS matrix
    dnp = Op.n;
    np = dnp / 2;
    I = find(incidenceFull);
    [nbnonzero] = size(I,1);
    S = incidenceFull;
    deltaS = objArray(np);
    for t = 1: np
        deltaS(t).value = S;
    end
    for i = 1:nbnonzero
        [k,a] = ind2sub(size(incidenceFull), I(i));
        col = find(incidenceFull(k,:));
        temp = 0;
        delta = zeros(np,1);
        for j = 1:size(col,1)
            for t = 1: np
                u = Atts(t).Value(k,j) - Atts(t).Value(k,a);
                temp = temp + log(1 + exp(Op.x(t) * u));
                delta(t) = delta(t) + (exp(Op.x(t) * u) * u)/(1 + exp(Op.x(t) * u));  
            end
        end
        S(k,a) = temp;
        for t = 1: np
            deltaS(t).value(k,a) = delta(t);
        end
    end 
    deltaSr = objArray(np); % Regular S
    gradZr = objArray(np);
    gradZu = objArray(np);
    for t = 1:np
        deltaSr(t).value =  deltaS(t).value(1:lastIndexNetworkState,1:lastIndexNetworkState);
        deltaSr(t).value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        deltaSr(t).value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end
    MrFull = incidenceFull .*  exp(-S);
    %% Mr and Sr for RRM
    MregularNetwork = MrFull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    Mr = MregularNetwork;
    Mr(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    Mr(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    SregularNetwork = S(1:lastIndexNetworkState,1:lastIndexNetworkState);
    Sr = SregularNetwork;
    Sr(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    Sr(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    
    %% Compute Mu,Uu
    MuFull = getM(Op.x(np+1:dnp), isLinkSizeInclusive);
    MregularNetwork = MuFull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    Ufull = getU(Op.x(np+1:dnp), isLinkSizeInclusive);
    UregularNetwork = Ufull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    % Initialize
    Mu = MregularNetwork;
    Mu(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    Mu(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    Uu = UregularNetwork;
    Uu(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    Uu(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    AttR = objArray(np);
    for i = 1:np
        AttR(i).value =  Atts(i).Value(1:lastIndexNetworkState,1:lastIndexNetworkState);
        AttR(i).value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        AttR(i).value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end
    GradLnP_RRM = zeros(1,np);
    GradLnP_RUM = zeros(1,np);
    %% Compute LL value  
    for n = 1:nbobs
        %n
        dest = Obs(n, 1);
        orig = Obs(n, 2);
        %% Regret - based::
        Mr(1:lastIndexNetworkState ,lastIndexNetworkState + 1) = MrFull(:,dest);
        [Zr, expVokBool] = getExpV(Mr); % vector with value functions for given beta                                                                     
        if (expVokBool == 0)
            disp('The parameters not fesible')
            return; 
        end          
        % Get delta Zr        
        for i = 1:np
            u = - Mr .* (deltaSr(i).value); 
            v = sparse(u * Zr); 
            A = speye(size(Mr)) - Mr;
            gradZr(i).value =  sparse(A\v); 
        end   
        %% Unitily-based
        Mu(1:lastIndexNetworkState ,lastIndexNetworkState + 1) = MuFull(:,dest);
        [Zu, expVokBool] = getExpV(Mu); % vector with value functions for given beta                                                                     
        if (expVokBool == 0)
            disp('The parameters not fesible')
            return; 
        end          
        % Get delta Zu        
        for i = 1:np
            u = Mu .* (AttR(i).value); 
            v = sparse(u * Zu); 
            A = speye(size(Mu)) - Mu;
            gradZu(i).value =  sparse(A\v); 
        end
        
        %% Compute the prob
        path = Obs(n,:);
        lpath = size(find(path),2);
        Gradient(n,:) =  zeros(1,Op.n);        
        for i = 2:lpath - 1
            maxLink = min(path(i+1),lastIndexNetworkState + 1);
            LnP_RRM = (-1/mu) * (S(path(i),path(i+1)) - log(Zr(maxLink)) + log(Zr(path(i))));            
            LnP_RUM = (1/mu) * (Ufull(path(i),path(i+1)) + log(Zu(maxLink)) - log(Zu(path(i))));
            for j = 1:np
                GradLnP_RRM(j) = (-1/mu) * ( deltaS(j).value(path(i),path(i+1)) - gradZr(j).value(maxLink)/Zr(maxLink) + gradZr(j).value(path(i))/Zr(path(i)));
                GradLnP_RUM(j) = (1/mu) * ( Atts(j).Value(path(i),path(i+1)) +  gradZu(j).value(maxLink)/Zu(maxLink) - gradZu(j).value(path(i))/Zu(path(i)));
            end
            if LnP_RRM > LnP_RUM
                Policy(n,i-1) = 1;
            else
                Policy(n,i-1) = 2;
            end
                 
        end        
    end   
end

