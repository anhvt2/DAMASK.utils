function zNext = getNextSamplingPointReduceMSE(gp_fic, S, Y, zLB, zUB);
	% define the global bounds for x at here
	% functions called: 
		% calcNegAcquis -> calcAcquis
		% cmaes

	% zLB = dmodel.zLB;
	% zUB = dmodel.zUB;

	[fBest,iBest] = min(Y);
	zBest = S(iBest,:);
	d = length(zBest);

	OPTS = cmaes; % get default options
	OPTS.LBounds = reshape(zLB, d, 1); % set lowerbounds
	OPTS.UBounds = reshape(zUB, d, 1); % set upperbounds
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
	for i = 1:d
		zInitCMAES(i) = zLB(i) + rand() * (zUB(i) - zLB(i));
	end
	zNext = cmaes('calculateNegativeMSE', zInitCMAES, sigma, OPTS, gp_fic, S, Y, zLB, zUB); 
	zNext = reshape(zNext, 1, d); % convert to row vector
	
end

