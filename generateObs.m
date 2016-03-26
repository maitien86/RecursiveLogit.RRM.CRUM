%   Generate Observatios with link size is include
%   The link size attributes are already computed for all pairs OD
%   --------
%   filename:   Name of file which stores the observations
%   x0:         Given parameters
%   ODpairs:    Matrix with all OD pairs
%   nbobsOD:    Number of generated obs each OD
%%
function ok = generateObs(filename, x0, ODpairs, nbobsOD)

    global incidenceFull; 
    global Mfull;
    global Ufull;
    global Atts;
    global Op;
    global isLinkSizeInclusive;
    
    % Generate Obs
    % ----------------------------------------------------  
    mu = 1; % MU IS NORMALIZED TO ONE
    % Parameter for the radom term
    location = 0;
    scale = 1;
    euler = 0.577215665;    
    [lastIndexNetworkState,maxDest] = size(incidenceFull);
    [p q] = size(incidenceFull);
    nbobs = size(ODpairs,1);
    % For the OD independence attributes
    if isLinkSizeInclusive == true
        sizeOfParams = Op.n - 1 
    else
        sizeOfParams = Op.n;
    end
      
    Mfull = getM(x0, isLinkSizeInclusive);
    MregularNetwork = Mfull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    Ufull = getU(x0, isLinkSizeInclusive);
    UregularNetwork = Ufull(1:lastIndexNetworkState,1:lastIndexNetworkState);
    % Set LL value
    LL = 0;
    grad = zeros(1, Op.n);
    % Initialize
    M = MregularNetwork;
    M(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    M(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    U = UregularNetwork;
    U(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
    U(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    for i = 1:Op.n
        AttLc(i) =  Matrix2D(Atts(i).Value(1:lastIndexNetworkState,1:lastIndexNetworkState));
        AttLc(i).Value(:,lastIndexNetworkState+1) = sparse(zeros(lastIndexNetworkState,1));
        AttLc(i).Value(lastIndexNetworkState+1,:) = sparse(zeros(1, lastIndexNetworkState + 1));
    end

    % b matrix:
    N = size(M,1);
    b = sparse(zeros(N,1));
    b(N) = 1;
    B = sparse(zeros(N, maxDest - lastIndexNetworkState));
    B(N,:) = ones(1,maxDest - lastIndexNetworkState);
    for i = 1: maxDest - lastIndexNetworkState
        B(1:lastIndexNetworkState,i) = Mfull(:, i+lastIndexNetworkState);
    end
    A = speye(size(M)) - M;
    Z = A\B;
    % Check feasible
    minele = min(Z(:));
    expVokBool = 1;
    if minele == 0 || minele < OptimizeConstant.NUM_ERROR
       expVokBool = 0;
    end 
    Zabs = abs(Z); % MAYBE SET TO VERY SMALL POSITIVE VALUE? 
    D = (A * Z - B);
    resNorm = norm(D(:));
    if resNorm > OptimizeConstant.RESIDUAL
       expVokBool = 0;
    end
    if (expVokBool == 0)
            LL = OptimizeConstant.LL_ERROR_VALUE;
            grad = ones(Op.n,1);
            disp('The parameters not fesible')
            return; 
    end
    
    IregularNetwork = incidenceFull(1:lastIndexNetworkState,1:lastIndexNetworkState);    
    dummy = lastIndexNetworkState + 1;
        
    % Obs = sparse(zeros(nbobs * nbobsOD, dummy));
    % Loop over all OD pairs
    for n = 1:nbobs
        n
        dest = ODpairs(n, 1);
        orig = ODpairs(n, 2);            
        addColumn = incidenceFull(:,dest);
        Incidence = IregularNetwork;
        Incidence(:,lastIndexNetworkState+1) = addColumn;
        expV = Z(:,dest - lastIndexNetworkState);
        expV = full(abs(expV));  
        V = log(expV);
        % Now we have all utilities, time to generate the observations      
        for i = 1: nbobsOD
            Obs((n-1)*nbobsOD + i, 1) = dest;
            Obs((n-1)*nbobsOD + i, 2) = orig;
            k = orig;
            t = 3;
            while k ~= dummy
                ind = find(Incidence(k,:));
                nbInd = size(ind,2);
                bestUtilities = -1e6;
                for j = 1: nbInd
                    utility = U(k,ind(j)) + V(ind(j)) + random('ev',location,scale) - euler ;                   
                    if bestUtilities < utility
                        bestInd = ind(j);
                        bestUtilities = utility;
                    end
                end
                if bestInd ~= dummy
                    Obs((n-1)*nbobsOD + i, t) = bestInd;
                    t = t + 1;
                end
                k = bestInd;
            end
            Obs((n-1)*nbobsOD + i, t) = dest;
        end
    end
    % Write to file::
    [i,j,val] = find(Obs);
    data_dump = [i,j,val];
    save(filename,'data_dump','-ascii');
    ok = true;
end