function xNext = getNextSamplingPointReduceMSE(dmodel)
	% define the global bounds for x at here
	% functions called: 
		% calcNegAcquis -> calcAcquis
		% cmaes

	xLB = dmodel.xLB;
	xUB = dmodel.xUB;

	[fBest,iBest] = min(dmodel.origY);
	xBest = dmodel.origS(iBest,:);
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
	xNext = cmaes('calculateNegativeMSE', xInitCMAES, sigma, OPTS, dmodel); 
	
end

