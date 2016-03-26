% MAIN
%%
Credits;
globalVar;
global resultsTXT; 

%------------------------------------------------------

file_linkIncidence = './Input/linkIncidence.txt';
file_AttEstimatedtime = './Input/TravelTime.txt';
file_turnAngles = './Input/TurnAngles.txt';
file_observations = './simulatedData/ObservationsAll.txt';

% Initialize the optimizer structure

isLinkSizeInclusive = false;
isFixedUturn = false;
loadData;
Op = Op_structure;
initialize_optimization_structure();
Op.x = [-2.494;-0.933;-1.0;-0.4;    2.494;0.933;1.0;0.4;   0.5;0.5;0.5;0.5];    


Op.Optim_Method = OptimizeConstant.TRUST_REGION_METHOD;
Op.Hessian_approx = OptimizeConstant.BHHH;
Gradient = zeros(nbobs,Op.n);

options_const =  optimoptions(@fmincon,'Display','iter','Algorithm','interior-point','GradObj','on');
lb = -100 * ones(Op.n,1);
ub = 100 * ones(Op.n,1);

lb(9:12) = 0;
ub(9:12) = 1;

disp('Start Optimizing ....')
[x,fval,exitflag,output,lambda,grad] = fmincon(@LL,Op.x,[],[],[],[],lb,ub,[],options_const);

%---------------------------
%Starting optimization
%{ tic ;
% disp('Start Optimizing ....')
% [Op.value, Op.grad ] = LL(Op.x);
% Op.delta = 0.1 * norm(Op.grad);
% %assignPolicy();
% PrintOut(Op);
% % print result to string text
% header = [sprintf('%s \n',file_observations) Op.Optim_Method];
% header = [header sprintf('\nNumber of observations = %d \n', nbobs)];
% header = [header sprintf('Hessian approx methods = %s \n', OptimizeConstant.getHessianApprox(Op.Hessian_approx))];
% resultsTXT = header;
% %------------------------------------------------
% while (true)    
%   Op.k = Op.k + 1;
%   if strcmp(Op.Optim_Method,OptimizeConstant.LINE_SEARCH_METHOD);
%     ok = line_search_iterate();
%     if ok == true
%         PrintOut(Op);
%     else
%         disp(' Unsuccessful process ...')
%         break;
%     end
%   else
%     ok = btr_interate();
%     PrintOut(Op);
%   end
%   [isStop, Stoppingtype, isSuccess] = CheckStopping(Op);  
%   %----------------------------------------
%   % Check stopping criteria
%   if(isStop == true)
%       isSuccess
%       fprintf('The algorithm stops, due to %s', Stoppingtype);
%       resultsTXT = [resultsTXT sprintf('The algorithm stops, due to %s \n', Stoppingtype)];
%       break;
%   end
% end


%%   Compute variance - Covariance matrix
disp(' Calculating VAR-COV ...');
getCov;

%% Finishing ...
ElapsedTtime = toc
resultsTXT = [resultsTXT sprintf('\n Number of function evaluation %d \n', Op.nFev)];
resultsTXT = [resultsTXT sprintf('\n Estimated time %d \n', ElapsedTtime)];
