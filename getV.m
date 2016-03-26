function [V, boolV0] = getV(M,B)
    N = size(M,1);   
    m = size(B,2);
    j = 0;
    V = zeros(size(B));
    while(1)
         j = j+1
         nextV = getNextV(M,B,V);
         residual = V - nextV;
         norm(residual)
         if norm(residual) < 0.0001;% norm(Op.grad)*10 %0.0001
             break;
         end
         V = nextV; 
         if (j > 200)             
            break ;
         end
    end
    % minele = min(V(:));
     boolV0 = 1;
     if minele == 0 || minele < OptimizeConstant.NUM_ERROR
       boolV0 = 0;
       fprintf('min zero');
     end    
     if residual > 10 || (~isreal(V))
       boolV0 = 0;
     end    
end

function nextV = getNextV(M,B,V)
    nextV = log(M * exp(V) + B); 
end
    