function xNext = getNextSamplingPoint(dmodel, dmodelClassif, acquisitionFunction, ooDaceOpts)
	% define the global bounds for x at here
	% functions called: 
		% calculateNegativeAcquisitionFunction -> calcAcquis
		% cmaes

	% acquisitionFunction: string like input: 'PI', 'EI', or 'UCB'

	[fBest,iBest] = max(dmodel.getValues());
	S = dmodel.getSamples();
	xBest = S(iBest,:);
	d = length(xBest);

	xLB = ooDaceOpts.xLB;
	xUB = ooDaceOpts.xUB;

	OPTS = cmaes; % get default options
	OPTS.LBounds = reshape(xLB, length(xLB), 1); % set lower bounds
	OPTS.UBounds = reshape(xUB, length(xUB), 1); % set upper bounds
	% OPTS.Restarts = 3;
	% run mode
	% OPTS.MaxIter = 5000; % set MaxIter
	% OPTS.MaxFunEvals = 5000; % set MaxFunEvals
	OPTS.SaveVariables = 'off'; % do not save .mat for CMA-ES
	% % debug mode
	% OPTS.MaxIter = 12; % set MaxIter
	% OPTS.MaxFunEvals = 20; % set MaxFunEvals
	% OPTS.DispModulo = 1; % debug: aggressive settings

	% compute sigma based on the thresholds
	d = length(xUB);
	sigma = 1/3  * reshape(xUB - xLB, d, 1);
	% sigma = 0.25 * min(xUB - xLB); 

	% NOTE: argmax instead of argmin in CMAES (need a -f)
	% d = length(xUB);
	xInitCMAES = xLB + rand(1,d) .* (xUB - xLB);
	% xInitCMAES = getGoodInitSamplingPoint(parentPath);
	% xNext = cmaes('calculateNegativeAcquisitionFunction', xInitCMAES, sigma, OPTS, dmodel); 
	% xNext = cmaes('calculateNegativeAcquisitionFunction', xBest, sigma, OPTS, dmodel); 
	% xNext = cmaes('calculateNegativeAcquisitionFunction', 0.5 * (xBest + xInitCMAES), sigma, OPTS, dmodel); 
	% xNext = cmaes('calculateNegativeAcquisitionFunction', w * xBest + (1 - w) * xInitCMAES, sigma, OPTS, dmodel); % original

	for i = 1:d
		xRand(i) = xLB(i) + rand() * (xUB(i) - xLB(i));
	end

	if strcmp(acquisitionFunction, 'MC')
		%%% Monte Carlo 
		fprintf('getNextSamplingPoint.m: Monte Carlo random sampling activated.\n')
		xNext = xRand;
		return;
	end

	w = rand(); 
	xNext = cmaes('calculateNegativeAcquisitionFunction', w * xBest + (1 - w) * xInitCMAES, sigma, OPTS, dmodel, dmodelClassif, acquisitionFunction);
	xNext = reshape(xNext, 1, d);
	constraintViolator = checkConstraint(xNext); % return 0 if violated; 1 if not
	multipleDesignSitesViolator = checkMultipleDesignSites(xNext, dmodel.getSamples()); % return 1 if duplicated, 0 if not
	fprintf('getNextSamplingPoint.m: constraintViolator = %d\n', constraintViolator);
	fprintf('getNextSamplingPoint.m: multipleDesignSitesViolator = %d\n', multipleDesignSitesViolator);

	% if (constraint is not violated) and (xNext is not duplicated)
	while ~constraintViolator || multipleDesignSitesViolator
		xInitCMAES = xLB + rand(1,d) .* (xUB - xLB);
		w = rand();
		xNext = cmaes('calculateNegativeAcquisitionFunction', w * xBest + (1 - w) * xInitCMAES, sigma, OPTS, dmodel, dmodelClassif, acquisitionFunction);
		xNext = reshape(xNext, 1, d);
		constraintViolator = checkConstraint(xNext);
		multipleDesignSitesViolator = checkMultipleDesignSites(xNext, dmodel.getSamples());
		fprintf('getNextSamplingPoint.m: constraintViolator = %d\n', constraintViolator);
		fprintf('getNextSamplingPoint.m: multipleDesignSitesViolator = %d\n', multipleDesignSitesViolator);
	end

end

