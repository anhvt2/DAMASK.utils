function [x,fval,exitFlag,output] = particleswarm(fun, nvars, lb, ub, userOptions)
%PARTICLESWARM Bound constrained optimization using particle swarm optimization
%
%   PARTICLESWARM attempts to solve problems of the form:
%       min F(X)  subject to  LB <= X <= UB
%        X
%
%   X = PARTICLESWARM(FUN,NVARS) finds a local unconstrained minimum X to
%   the objective function FUN. NVARS is the dimension (number of design
%   variables) of FUN. FUN accepts a vector X of size 1-by-NVARS and
%   returns a scalar evaluated at X.
%
%   X = PARTICLESWARM(FUN,NVARS,LB,UB) defines a set of lower and upper
%   bounds on the design variables, X, so that a solution is found in the
%   range LB <= X <= UB. Use empty matrices for LB and UB if no bounds
%   exist. Set LB(i) = -Inf if X(i) is unbounded below; set UB(i) = Inf if
%   X(i) is unbounded above.
%
%   X = PARTICLESWARM(FUN,NVARS,LB,UB,OPTIONS) minimizes with the default
%   optimization parameters replaced by values in OPTIONS, an argument
%   created with the OPTIMOPTIONS function. See OPTIMOPTIONS for details.
%   For a list of options accepted by PARTICLESWARM refer to the
%   documentation.
%
%   X = PARTICLESWARM(PROBLEM) finds the minimum for PROBLEM. PROBLEM is a
%   structure that has the following fields:
%       objective: <Objective function>
%           nvars: <Number of design variables>
%              lb: <Lower bound on X>
%              ub: <Upper bound on X>
%         options: <Options created with optimoptions('particleswarm',...)>
%        rngstate: <State of the random number generator>
%
%   [X,FVAL] = PARTICLESWARM(FUN, ...) returns FVAL, the value of the
%   objective function FUN at the solution X.
%
%   [X,FVAL,EXITFLAG] = PARTICLESWARM(FUN, ...) returns EXITFLAG which
%   describes the exit condition of PARTICLESWARM. Possible values of
%   EXITFLAG and the corresponding exit conditions are
%
%     1 Relative change in best value of the objective function over
%        OPTIONS.MaxStallIterations iterations less than OPTIONS.FunctionTolerance.
%     0 Maximum number of iterations exceeded.
%    -1 Optimization ended by the output or plot function.
%    -2 The bounds are inconsistent (due to lower bounds exceeding upper
%        bounds).
%    -3 OPTIONS.ObjectiveLimit reached.
%    -4 Best known value of the objective function did not change within 
%        the stall time limit (as specified in OPTIONS.MaxStallTime,
%        in seconds).
%    -5 The total time of the optimization exceeded the time limit (as 
%        specified in OPTIONS.MaxTime, in seconds).
%
%   [X,FVAL,EXITFLAG,OUTPUT] = PARTICLESWARM(FUN, ...) returns a structure
%   OUTPUT with the following information:
%      iterations: Total iterations
%       funccount: The number of evaluations of the objective function.
%         message: The reason the algorithm ended.
%        rngstate: State of the MATLAB random number generator, just before
%                  the algorithm started.
%
%   Examples:
%     Unconstrained minimization of Dejong's fifth function of 2 variables
%      fcn = @dejong5fcn;
%      nvars = 2;
%      [x,fval] = particleswarm(fcn,nvars)
%
%     Minimization of Dejong's fifth function of 2 variables, subject to
%     lower and upper bounds:
%      fcn = @dejong5fcn;
%      nvars = 2;
%      lb = [-64 -64];
%      ub = [64 64];
%      [x,fval] = particleswarm(fcn,nvars,lb,ub)
% 
%     Minimization of Dejong's fifth function of 2 variables, with
%     iterative display:
%      fcn = @dejong5fcn;
%      nvars = 2;
%      options = optimoptions('particleswarm', 'Display', 'iter');
%      [x,fval] = particleswarm(fcn,nvars,[],[],options)
%
%
%   See also OPTIMOPTIONS, PATTERNSEARCH, GA, FMINSEARCH, @.

%   Copyright 2012-2016 The MathWorks, Inc.

% Check the number of input arguments
narginchk(1,5);

% One input argument is for problem structure
if nargin == 1
    if isa(fun,'struct')
        [fun,nvars,lb,ub,rngstate,userOptions] = separateOptimStruct(fun);
        % Reset the random number generators
        resetDfltRng(rngstate);
    else % Single input and non-structure. 
        error(message('globaloptim:particleswarm:invalidStructInput'));
    end
elseif nargin < 5
    % Define optional arguments that were omitted.
    userOptions = [];
    if nargin < 4
        ub = [];
        if nargin < 3
            lb = [];
        end
    end
end

[x,fval] = deal([]);

output = struct('rngstate', rng, 'iterations', 0, 'funccount', 0, 'message', '');

% Arg fun
validateattributes(fun, {'function_handle'}, {'scalar'}, 'particleswarm', 'fun');

% Arg nvars
validateattributes(nvars, {'numeric'}, {'scalar', 'positive', 'integer'}, 'particleswarm', 'nvars');

% Ensure all numeric inputs are doubles
msg = isoptimargdbl('PARTICLESWARM', {'NVARS', 'lb', 'ub'}, nvars, lb, ub);
if ~isempty(msg)
    error('globaloptim:particleswarm:dataType', msg);
end

% Handle options before lb/ub, because the Verbosity option may be needed.
if isempty(userOptions)
    userOptions = optimoptions('particleswarm'); % default options
end
[options, objFcn] = validateOptions(userOptions, nvars, fun, lb, ub);

% Args lb & ub
[lbColumn, ubColumn, msg, exitFlag] = checkbound(lb, ub, nvars);
% checkbound returns column vectors, but we to convert that to a row vector
% and repeat it across SwarmSize rows.
lbRow = lbColumn';
ubRow = ubColumn';
if exitFlag < 0
    output.message = msg; % This is the first message, so no need to append
    if options.Verbosity > 0
        fprintf('%s\n', msg);
    end
    return
end

% Perform check on initial population, fvals, and range
options = initialParticleCheck(options);

% Start the core algorithm
[x,fval,exitFlag,output] = pswcore(objFcn,nvars,lbRow,ubRow,output,options);
end

function [x,fval,exitFlag,output] = pswcore(objFcn,nvars,lbRow,ubRow,output,options)
exitFlag=[];

% Get algorithmic options
numParticles = options.SwarmSize;
isVectorized = strcmp(options.Vectorized, 'on');
cSelf = options.SelfAdjustment;
cSocial = options.SocialAdjustment;
minNeighborhoodSize = max(2,floor(numParticles*options.MinFractionNeighbors));
minInertia = options.InertiaRange(1);
maxInertia = options.InertiaRange(2);
lbMatrix = repmat(lbRow, numParticles, 1);
ubMatrix = repmat(ubRow, numParticles, 1);

% Create initial state: particle positions & velocities, fvals, status data
state = makeState(nvars,lbMatrix,ubMatrix,objFcn,options);
bestFvals = min(state.Fvals);
% Create a vector to store the last StallIterLimit bestFvals.
% bestFvalsWindow is a circular buffer, so that the value from the i'th
% iteration is stored in element with index mod(i-1,StallIterLimit)+1.
bestFvalsWindow = nan(options.StallIterLimit, 1);

% Initialize adaptive parameters:
%   initial inertia = maximum *magnitude* inertia
%   initial neighborhood size = minimum neighborhood size
adaptiveInertiaCounter = 0;
if all(options.InertiaRange >= 0)
    adaptiveInertia = maxInertia;
elseif all(options.InertiaRange <= 0)
    adaptiveInertia = minInertia;
else
    % checkfield should prevent InertiaRange from having positive and
    % negative vlaues.
    assert(false, 'globaloptim:particleswarm:invalidInertiaRange', ...
        'The InertiaRange option should not contain both positive and negative numbers.');
end
adaptiveNeighborhoodSize = minNeighborhoodSize;

% Output functions.
if isempty(options.OutputFcns)
    haveoutputfcn = false;
else
    haveoutputfcn = true;
    % Parse options.OutputFcns which is needed to support cell array
    % syntax.
     options.OutputFcns = createCellArrayOfFunctions(options.OutputFcns,'OutputFcns');
end

% Plot functions.
if isempty(options.PlotFcns)
    haveplotfcn = false;
else
    haveplotfcn = true;
    % Parse options.PlotFcns which is needed to support cell array
    % syntax.
    options.PlotFcns = createCellArrayOfFunctions(options.PlotFcns,'PlotFcns');
end

% Setup display header 
if  options.Verbosity > 1
    fprintf('\n                                 Best            Mean     Stall\n');
    fprintf(  'Iteration     f-count            f(x)            f(x)    Iterations\n');
    fprintf('%5.0f         %7.0f    %12.4g    %12.4g    %5.0f\n', ...
        0, state.FunEval, bestFvals, mean(state.Fvals), 0);
end

% Allow output and plot functions to perform any initialization tasks
if haveoutputfcn || haveplotfcn
    % For calling an OutputFcn, make an options object (to be updated
    % later) that can be passed in
    options.OutputPlotFcnOptions  = optimoptions(@particleswarm);
    options.OutputPlotFcnOptions  = copyForOutputAndPlotFcn(options.OutputPlotFcnOptions,options);
    
    optimValues = i_updateOptimValues;
    state.StopFlag = callOutputAndPlotFcns(options, optimValues, 'init', 'particleswarm');
    % check to see if any stopping criteria have been met
    [exitFlag, output.message] = stopParticleswarm(options,state,bestFvalsWindow);
else
    state.StopFlag = false;
end

% Run the main loop until some exit condition becomes true
while isempty(exitFlag)
        state.Iteration = state.Iteration + 1;

        % Generate a random neighborhood for each particle that includes
        % the particle itself
        neighborIndex = zeros(numParticles, adaptiveNeighborhoodSize);
        neighborIndex(:, 1) = 1:numParticles; % First neighbor is self
        for i = 1:numParticles
            % Determine random neighbors that exclude the particle itself,
            % which is (numParticles-1) particles
            neighbors = randperm(numParticles-1, adaptiveNeighborhoodSize-1);
            % Add 1 to indicies that are >= current particle index
            iShift = neighbors >= i;
            neighbors(iShift) = neighbors(iShift) + 1;
            neighborIndex(i,2:end) = neighbors;
        end
        % Identify the best neighbor
        [~, bestRowIndex] = min(state.IndividualBestFvals(neighborIndex), [], 2);
        % Create the linear index into neighborIndex
        bestLinearIndex = (bestRowIndex.'-1).*numParticles + (1:numParticles);
        bestNeighborIndex = neighborIndex(bestLinearIndex);
        randSelf = rand(numParticles, nvars);
        randSocial = rand(numParticles, nvars);

        % Note that velocities and positions can become infinite if the
        % inertia range is too large or if the objective function is badly
        % behaved.

        % Update the velocities
        newVelocities = adaptiveInertia*state.Velocities + ...
            cSelf*randSelf.*(state.IndividualBestPositions-state.Positions) + ...
            cSocial*randSocial.*(state.IndividualBestPositions(bestNeighborIndex, :)-state.Positions);
        tfValid = all(isfinite(newVelocities), 2);
        state.Velocities(tfValid,:) = newVelocities(tfValid,:);
        % Update the positions
        newPopulation = state.Positions + state.Velocities;
        tfInvalid = ~isfinite(newPopulation);
        newPopulation(tfInvalid) = state.Positions(tfInvalid);
        % Enforce bounds, setting the corresponding velocity component to
        % zero if a particle encounters a lower/upper bound
        tfInvalid = newPopulation < lbMatrix;
        if any(tfInvalid(:))
            newPopulation(tfInvalid) = lbMatrix(tfInvalid);
            state.Velocities(tfInvalid) = 0;
        end
        tfInvalid = newPopulation > ubMatrix;
        if any(tfInvalid(:))
            newPopulation(tfInvalid) = ubMatrix(tfInvalid);
            state.Velocities(tfInvalid) = 0;
        end
        state.Positions = newPopulation;
        
        if isVectorized
            state.Fvals = objFcn(state.Positions);
        else
            state.Fvals = fcnvectorizer(state.Positions, objFcn, 1, options.SerialUserFcn);
        end
        state.FunEval = state.FunEval + numParticles;

        % Remember the best fvals and positions
        tfImproved = state.Fvals < state.IndividualBestFvals;
        state.IndividualBestFvals(tfImproved) = state.Fvals(tfImproved);
        state.IndividualBestPositions(tfImproved, :) = state.Positions(tfImproved, :);
        bestFvalsWindow(1+mod(state.Iteration-1,options.StallIterLimit)) = min(state.IndividualBestFvals);
        
        % Keep track of improvement in bestFvals and update the adaptive
        % parameters according to the approach described in S. Iadevaia et
        % al. Cancer Res 2010;70:6704-6714 and M. Liu, D. Shin, and H. I.
        % Kang. International Conference on Information, Communications and
        % Signal Processing 2009:1-5.

        newBest = min(state.IndividualBestFvals);
        if isfinite(newBest) && newBest < bestFvals
            bestFvals = newBest;
            state.LastImprovement = state.Iteration;
            state.LastImprovementTime = toc(state.StartTime);
            adaptiveInertiaCounter = max(0, adaptiveInertiaCounter-1);
            adaptiveNeighborhoodSize = minNeighborhoodSize;
        else
            adaptiveInertiaCounter = adaptiveInertiaCounter+1;
            adaptiveNeighborhoodSize = min(numParticles, adaptiveNeighborhoodSize+minNeighborhoodSize);
        end
        
        % Update the inertia coefficient, enforcing limits (Since inertia
        % can be negative, enforcing both upper *and* lower bounds after
        % multiplying.)
        if adaptiveInertiaCounter < 2
            adaptiveInertia = max(minInertia, min(maxInertia, 2*adaptiveInertia));
        elseif adaptiveInertiaCounter > 5
            adaptiveInertia = max(minInertia, min(maxInertia, 0.5*adaptiveInertia));
        end
        
        % Call output and plot functions
        if haveoutputfcn || haveplotfcn
            optimValues = i_updateOptimValues;
            state.StopFlag = callOutputAndPlotFcns(options, optimValues, 'iter', 'particleswarm');
        else
            state.StopFlag = false;
        end

        % check to see if any stopping criteria have been met
        [exitFlag, output.message] = stopParticleswarm(options,state,bestFvalsWindow);
end % End while loop

% Find and return the best solution
[fval,indexBestFval] = min(state.IndividualBestFvals);
x = state.IndividualBestPositions(indexBestFval,:);

% Update output structure
output.iterations = state.Iteration;
output.funccount   = state.FunEval;

% Allow output and plot functions to perform any clean up tasks.
if haveoutputfcn || haveplotfcn
    optimValues = i_updateOptimValues;
    callOutputAndPlotFcns(options, optimValues, 'done', 'particleswarm');
end

% A hybrid scheme. Try another minimization method if there is one.
if ~isempty(options.HybridFcn)
    [xhybrid,fhybrid,output] = callHybridFunction(objFcn, x, lbRow, ubRow, options, output);
    if ~isempty(fhybrid) && fhybrid < fval
        fval = fhybrid;
        x = xhybrid;
    end
end

% Nested function
    function optimValues = i_updateOptimValues
        %updateOptimValues update the structure to be passed to user functions
        % Prepare data to be sent over to output/plot functions
        [optimValues.bestfval,indBestFval] = min(state.IndividualBestFvals);        
        optimValues.bestx = state.IndividualBestPositions(indBestFval,:);
        optimValues.iteration = state.Iteration;
        optimValues.funccount = state.FunEval;
        optimValues.meanfval = meanf(state.Fvals);
        optimValues.stalliterations = max(0,state.Iteration  - state.LastImprovement);
        optimValues.swarm = state.Positions;    
        optimValues.swarmfvals = state.Fvals;         
    end
end % End function pswcore

function [xhybrid,fhybrid,output] = callHybridFunction(objFcn, x, lb, ub, options, output)
% Determine the hybrid function
if isa(options.HybridFcn,'function_handle')
    hfunc = func2str(options.HybridFcn);
else
    hfunc = options.HybridFcn;
end

% Inform about hybrid scheme
if  options.Verbosity > 1
    fprintf('%s%s%s\n','Switching to the hybrid optimization algorithm (',upper(hfunc),').');
end

[xhybrid,fhybrid,funccount,theMessage] = ...
    callHybrid(hfunc,objFcn,x,options.HybridFcnArgs,[],[],[],[],lb,ub);
output.funccount = output.funccount + funccount;
output.message   = sprintf([output.message '\n', theMessage '\n']);

% Inform about hybrid scheme ending
if  options.Verbosity > 1
    fprintf('%s%s\n',upper(hfunc), ' ended.');
end
end % End of callHybridFunction

function [exitFlag,reasonToStop] = stopParticleswarm(options,state,bestFvalsWindow)
iteration = state.Iteration;

iterationIndex = 1+mod(iteration-1,options.StallIterLimit);
bestFval = bestFvalsWindow(iterationIndex);
if options.Verbosity > 1 && ...
        mod(iteration,options.DisplayInterval)==0 && ... 
            iteration > 0
    FunEval  = state.FunEval;
    MeanFval = meanf(state.Fvals);
    StallGen = iteration  - state.LastImprovement;
    fprintf('%5.0f         %7.0f    %12.4g    %12.4g    %5.0f\n', ...
        iteration, FunEval, bestFval, MeanFval, StallGen);
end

% Compute change in fval and individuals in last 'Window' iterations
Window = options.StallIterLimit;
if iteration > Window
    % The smallest fval in the window should be bestFval.
    % The largest fval in the window should be the oldest one in the
    % window. This value is at iterationIndex+1 (or 1).
    if iterationIndex == Window
        % The window runs from index 1:iterationIndex
        maxBestFvalsWindow = bestFvalsWindow(1);
    else
        % The window runs from [iterationIndex+1:end, 1:iterationIndex]
        maxBestFvalsWindow = bestFvalsWindow(iterationIndex+1);
    end
    funChange = abs(maxBestFvalsWindow-bestFval)/max(1,abs(bestFval));
else
    funChange = Inf;
end

reasonToStop = '';
exitFlag = [];
if state.Iteration >= options.MaxIter
    reasonToStop = getString(message('globaloptim:particleswarm:ExitMaxIter'));
    exitFlag = 0;
elseif toc(state.StartTime) > options.MaxTime
    reasonToStop = getString(message('globaloptim:particleswarm:ExitMaxTime'));
    exitFlag = -5;
elseif (toc(state.StartTime)-state.LastImprovementTime) > options.StallTimeLimit
    reasonToStop = getString(message('globaloptim:particleswarm:ExitStallTimeLimit'));
    exitFlag = -4;
elseif bestFval <= options.ObjectiveLimit
    reasonToStop = getString(message('globaloptim:particleswarm:ExitObjectiveLimit'));
    exitFlag = -3;
elseif state.StopFlag
    reasonToStop = getString(message('globaloptim:particleswarm:ExitOutputPlotFcn'));
    exitFlag = -1;
elseif funChange <= options.TolFunValue
    reasonToStop = getString(message('globaloptim:particleswarm:ExitTolFun'));    
    exitFlag = 1;
end

if ~isempty(reasonToStop) && options.Verbosity > 0
    fprintf('%s\n',reasonToStop);
    return
end

% Print header again
if options.Verbosity > 1 && rem(iteration,30*options.DisplayInterval)==0 && iteration > 0
    fprintf('\n                                 Best            Mean     Stall\n');
    fprintf('Iteration     f-count            f(x)            f(x)    Iterations\n');
end

end

function m = meanf(x)
tfValid = ~isnan(x);
n = sum(tfValid);
if n==0
    % prevent divideByZero warnings
    m = NaN;
else
    % Sum up non-NaNs, and divide by the number of non-NaNs.
    m = sum(x(tfValid)) ./ n;
end
end % End of StopParticleswarm

function [options,objfcn]  = validateOptions(userOptions,nvars,objfcn,lb,ub)
% Most validation should have been performed by optim.options.checkfield.
% Additional validation includes:
%   * Validate options that depend on input arguments, like nvars.
%   * Validate options that depend on other options, like SwarmSize.
%   * Convert to numbers options that are strings.

if ~isa(userOptions, 'optim.options.SolverOptions')
    error(message('globaloptim:particleswarm:invalidOptionsInputs'));
end
userOptions = convertForSolver(userOptions, 'Particleswarm');
options = extractOptionsStructure(userOptions);

% Overwrite the default creation function for deployment.
if strcmp(func2str(isFcn(options.CreationFcn)),'pswcreationuniform')
    options.CreationFcn = @pswcreationuniform;
end

% Determine the verbosity
switch  options.Display
    case {'off','none'}
        options.Verbosity = 0;
    case 'final'
        options.Verbosity = 1;
    case 'iter'
        options.Verbosity = 2;
end

% SwarmSize (see optim options checkfield.m and Particleswarm.m)
if ischar(options.SwarmSize)
    % This should only be the magic string 'min(100,10*numberofvariables)'
    options.SwarmSize = min(100,10*nvars);
end

% MaxIter has several hard-coded string values that match the following
% format string (as enforced in checkfield.m):
nvarFormatStr = '%d*numberofvariables';

% MaxIter validation
if ischar(options.MaxIter)
    % Assumes string follows nvarFormatStr (as enforced by checkfield)
    options.MaxIter = nvars*sscanf(options.MaxIter, nvarFormatStr);
end

% Convert UseParallel to a boolean stored in SerialUserFcn
options.SerialUserFcn = ~validateopts_UseParallel(options.UseParallel,true,true);

% CreationFcn - only the function handle no additional arguments
options.CreationFcn = functionHandleOrCell('CreationFcn',options.CreationFcn);

% HybridFcn
if ~isempty(options.HybridFcn)
    [options.HybridFcn,options.HybridFcnArgs] = functionHandleOrCell('HybridFcn',options.HybridFcn);
    hybridFcnName = func2str(options.HybridFcn);
    
    unconstrainedHybridFcns = {'fminsearch','fminunc','patternsearch'};
    constrainedHybridFcns = {'fmincon','patternsearch'};
    allHybridFcns = union(unconstrainedHybridFcns,constrainedHybridFcns);

    stringSet('HybridFcn',hybridFcnName,allHybridFcns);
       
    bounded = any(isfinite(lb)) || any(isfinite(ub));
        
    % Check for a valid hybrid function for constrained problems
    if bounded && ~any(strcmpi(hybridFcnName,constrainedHybridFcns))
        msg = getString(message('globaloptim:validate:NotConstrainedHybridFcn', ...
                upper(hybridFcnName),strjoin(upper(constrainedHybridFcns),', ')));
        error('globaloptim:particleswarm:NotConstrainedHybridFcn',msg);
    elseif ~bounded && ~any(strcmpi(hybridFcnName,unconstrainedHybridFcns))
        msg = getString(message('globaloptim:validate:NotUnconstrainedHybridFcn', ...
                upper(hybridFcnName),strjoin(upper(unconstrainedHybridFcns),', ')));
        error('globaloptim:particleswarm:NotUnconstrainedHybridFcn',msg);
    end
    
    % If the user has set a hybrid function, they can specify options for the
    % hybrid function. If a user has passed a SolverOptions object for these
    % options, convert the options object to a structure. Note that we will not
    % warn here if a user specifies a solver with a different solver's options.
    if ~isempty(options.HybridFcnArgs) ...
            && isa(options.HybridFcnArgs{1}, 'optim.options.SolverOptions')
        % It is possible for a user to pass in a vector of options to the
        % solver. Silently use the first element in this array.
        options.HybridFcnArgs{1} = options.HybridFcnArgs{1}(1);
        
        % Extract the options structure
        options.HybridFcnArgs{1} = extractOptionsStructure(options.HybridFcnArgs{1});
    end
end

% ObjFcn
if ~isempty(objfcn)
    [objfcn,objFcnArgs] = functionHandleOrCell('ObjFcn',objfcn);
    if ~isempty(objFcnArgs)
        % Only need to create the anonymous function if there are
        % extra arguments.
        objfcn = createAnonymousFcn(objfcn,objFcnArgs);
    end
    % If necessary, create a function handle that enforces function value
    % checks.
    if strcmpi(options.FunValCheck, 'on')
        objfcnCell = optimfcnchk(objfcn,'particleswarm',nvars,true);
        objfcn = objfcnCell{3};
    end
else
    % This should only happen when called by psooutput
    objfcn = [];
end

% InitialSwarmSpan
options.InitialSwarmSpan = rangeCorrection(nvars,options.InitialSwarmSpan);

% Make sure that initial particles are consistent with nvars
if ~isempty(options.InitialSwarm) && size(options.InitialSwarm,2) ~= nvars
    error(message('globaloptim:particleswarm:wrongSizeInitialSwarm'));
end
end % validateOptions

function range = rangeCorrection(nvars,range)
%rangeCorrection Check the size of a range variable
range = reshape(range, 1, []); % Want row vector

% Perform scalar expansion, if required
if isscalar(range)
    range = repmat(range, 1, nvars);
end

% Check vector size
if ~isvector(range) || numel(range) ~= nvars
    error(message('globaloptim:particleswarm:invalidInitialSwarmSpan'));  
end

% Check for inf/nan range
if ~all(isfinite(range)) 
   error(message('globaloptim:particleswarm:nonFiniteInitialSwarmSpan'));
end
end % range correction

function options = initialParticleCheck(options)
% checkfield already checks that InitialSwarm is a matrix with nvars
% columns

numInitPositions = size(options.InitialSwarm, 1);

% No tests if initial positions is empty
if numInitPositions == 0
    return
end

% Warn if too many positions were specified.
numParticles  = options.SwarmSize;
if numInitPositions > numParticles
    warning(message('globaloptim:particleswarm:initialSwarmLength'));
    options.InitialSwarm(numParticles+1:numInitPositions,:) = [];
end

end % function initialParticleCheck

function state = makeState(nvars,lbMatrix,ubMatrix,objFcn,options)
% Create an initial set of particles and objective function values

% makeState needs the vector of bounds, not the expanded matrix.
lb = lbMatrix(1,:);
ub = ubMatrix(1,:);

% A variety of data used in various places
state = struct;
state.Iteration = 0; % current generation counter
state.StartTime = tic; % tic identifier
state.StopFlag = false; % OutputFcns flag to end the optimization
state.LastImprovement = 1; % generation stall counter
state.LastImprovementTime = 0; % stall time counter
state.FunEval = 0;
numParticles = options.SwarmSize;

% If InitialSwarm is partly empty use the creation function to generate
% population (CreationFcn can utilize InitialSwarm)
if numParticles ~= size(options.InitialSwarm,1)
    problemStruct = createProblemStruct('particleswarm');
    problemStruct.objective = objFcn;
    problemStruct.lb = lb;
    problemStruct.ub = ub;
    problemStruct.nvars = nvars;  
    problemStruct.options = options;
    if nargin(options.CreationFcn) ~= 1
        error(message('globaloptim:particleswarm:IncorrectNumInput',func2str(options.CreationFcn)))
    end    
    state.Positions = feval(options.CreationFcn,problemStruct);
else % the initial swarm was passed in!
    state.Positions = options.InitialSwarm;
end

% Enforce bounds
if any(any(state.Positions < lbMatrix)) || any(any(state.Positions > ubMatrix))
    state.Positions = max(lbMatrix, state.Positions);
    state.Positions = min(ubMatrix, state.Positions);
    if options.Verbosity > 1
        fprintf(getString(message('globaloptim:particleswarm:shiftX0ToBnds')));
    end
end

% Initialize velocities by randomly sampling over the smaller of 
% options.InitialSwarmSpan or ub-lb. Note that min will be
% InitialSwarmSpan if either lb or ub is not finite.
vmax = min(ub-lb, options.InitialSwarmSpan);
state.Velocities = repmat(-vmax,numParticles,1) + ...
    repmat(2*vmax,numParticles,1) .* rand(numParticles,nvars);

% Calculate the objective function for all particles.
objFcnErrMsgId = 'globaloptim:particleswarm:objectiveFcnFailed';
if strcmp(options.Vectorized, 'off')
    % Non-vectorized call to objFcn
    try
        firstFval = objFcn(state.Positions(1,:));
    catch userFcn_ME
        msg = message(objFcnErrMsgId);
        psw_ME = MException(msg.Identifier, getString(msg));
        userFcn_ME = addCause(userFcn_ME, psw_ME);
        rethrow(userFcn_ME)
    end
    % User-provided objective function should return a scalar
    if numel(firstFval) ~= 1
        error(message('globaloptim:particleswarm:objectiveCheck'));
    end        
    fvals = fcnvectorizer(state.Positions(2:end,:),objFcn,1,options.SerialUserFcn);
    % Concatenate the fvals of the first particle to the rest
    state.Fvals = [firstFval; fvals];
else
    % Vectorized call to objFcn
    try
        fvals = objFcn(state.Positions);
    catch userFcn_ME
        msg = message(objFcnErrMsgId);
        psw_ME = MException(msg.Identifier, getString(msg));
        userFcn_ME = addCause(userFcn_ME, psw_ME);
        rethrow(userFcn_ME)
    end
    if numel(fvals) ~= numParticles
        error(message('globaloptim:particleswarm:objectiveCheckVectorized'));
    end
    state.Fvals = fvals(:);
end
state.FunEval = numParticles;

state.IndividualBestFvals = state.Fvals;
state.IndividualBestPositions = state.Positions;
end % function makeState
