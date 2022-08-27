function xNext = getNextSamplingPointClfReduceMSE(hyp, meanfunc, covfunc, likfunc, hypClf, meanfuncClf, covfuncClf, likfuncClf, acquisitionFunction, xLB, xUB, S, Y, F)
	% define the global bounds for x at here
	% functions called: 
		% calcNegAcquis -> calcAcquis
		% cmaes

	% xLB = xLB; % trivial
	% xUB = xUB; % trivial

	[fBest,iBest] = min(Y);
	xBest = S(iBest,:);
	l = length(xBest);

	OPTS = cmaes; % get default options
	OPTS.LBounds = reshape(xLB, length(xLB), 1); % set lowerbounds
	OPTS.UBounds = reshape(xUB, length(xUB), 1); % set upperbounds
	OPTS.Restarts = 3;


	% run mode
	OPTS.MaxIter = 100; % set MaxIter
	OPTS.MaxFunEvals = 100; % set MaxFunEvals
	OPTS.SaveVariables = 'off'; % do not save .mat for CMA-ES
	% % debug mode
	% OPTS.MaxIter = 12; % set MaxIter
	% OPTS.MaxFunEvals = 20; % set MaxFunEvals
	% OPTS.DispModulo = 1; % debug: aggressive settings

	% compute sigma based on the thresholds
	sigma = 1; 

	% NOTE: argmax instead of argmin in CMAES (need a -f)
	for i = 1:l
		xInitCMAES(i) = xLB(i) + rand() * (xUB(i) - xLB(i));
	end
	xNext = cmaes('calculateNegativeClfMSE', xInitCMAES, sigma, OPTS, hypClf, meanfuncClf, covfuncClf, likfuncClf, S, Y, F); 
	xNext = reshape(xNext, 1, length(xNext));
end

