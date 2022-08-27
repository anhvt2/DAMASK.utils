function zNext = getNextSamplingPoint(gp_fic, S, Y, zLB, zUB, acquisitionFunction); % batch acquisition
	% define the global bounds for x at here
	% functions called: 
		% calculateNegativeAcquisitionFunction -> calcAcquis
		% cmaes

	% acquisitionFunction: string like input: 'PI', 'EI', or 'UCB'

	[fBest,iBest] = max(Y);
	zBest = S(iBest,:);
	l = length(zBest);

	OPTS = cmaes; % get default options
	d = length(zUB); % dimensionality
	OPTS.LBounds = reshape(zLB, d, 1); % set lower bounds
	OPTS.UBounds = reshape(zUB, d, 1); % set upper bounds
	% OPTS.Restarts = 3;
	% run mode
	OPTS.MaxIter = 500; % set MaxIter
	OPTS.MaxFunEvals = 500; % set MaxFunEvals
	OPTS.SaveVariables = 'off'; % do not save .mat for CMA-ES
	% % debug mode
	% OPTS.MaxIter = 12; % set MaxIter
	% OPTS.MaxFunEvals = 20; % set MaxFunEvals
	% OPTS.DispModulo = 1; % debug: aggressive settings

	% compute sigma based on the thresholds
	sigma = 1/3  * reshape(zUB - zLB, d, 1);
	% sigma = 0.25 * min(zUB - zLB); 

	% NOTE: argmax instead of argmin in CMAES (need a -f)
	% d = length(zUB);
	zInitCMAES = zLB + rand(1, d) .* (zUB - zLB);
	% zInitCMAES = getGoodInitSamplingPoint(parentPath);
	% zNext = cmaes('calculateNegativeAcquisitionFunction', zInitCMAES, sigma, OPTS, gp_fic); 
	% zNext = cmaes('calculateNegativeAcquisitionFunction', zBest, sigma, OPTS, gp_fic); 
	% zNext = cmaes('calculateNegativeAcquisitionFunction', 0.5 * (zBest + zInitCMAES), sigma, OPTS, gp_fic); 
	% zNext = cmaes('calculateNegativeAcquisitionFunction', w * zBest + (1 - w) * zInitCMAES, sigma, OPTS, gp_fic); % original

	for i = 1:l
		zRand(i) = zLB(i) + rand() * (zUB(i) - zLB(i));
	end

	if strcmp(acquisitionFunction, 'MC')
		%%% Monte Carlo 
		fprintf('getNextSamplingPoint.m: Monte Carlo random sampling activated.\n')
		zNext = zRand;
		return;
	end

	w = rand(); 
	zNext = cmaes('calculateNegativeAcquisitionFunction', w * zBest + (1 - w) * zInitCMAES, sigma, OPTS, gp_fic, S, Y, zLB, zUB, acquisitionFunction);
	constraintViolator = checkConstraint(zNext); % return 0 if violated; 1 if not
	multipleDesignSitesViolator = checkMultipleDesignSites(zNext, S); % return 1 if duplicated, 0 if not
	fprintf('getNextSamplingPoint.m: constraintViolator = %d\n', constraintViolator);
	fprintf('getNextSamplingPoint.m: multipleDesignSitesViolator = %d\n', multipleDesignSitesViolator);
	zNext = reshape(zNext, 1, d); % convert to row vector

	counter = 0;
	counterLimit = 5; % number of times needed to maximize the acquisition function

	% if (constraint is not violated) and (zNext is not duplicated)
	while ~constraintViolator || multipleDesignSitesViolator
		if counter < counterLimit
			zInitCMAES = zLB + rand(1,d) .* (zUB - zLB);
			w = rand();
			zNext = cmaes('calculateNegativeAcquisitionFunction', w * zBest + (1 - w) * zInitCMAES, sigma, OPTS, gp_fic, S, Y, zLB, zUB, acquisitionFunction);
			constraintViolator = checkConstraint(zNext);
			multipleDesignSitesViolator = checkMultipleDesignSites(zNext, S);
			fprintf('getNextSamplingPoint.m: constraintViolator = %d\n', constraintViolator);
			fprintf('getNextSamplingPoint.m: multipleDesignSitesViolator = %d\n', multipleDesignSitesViolator);
			zNext = reshape(zNext, 1, d); % convert to row vector
		else
			zNext = w * zBest + (1 - w) * zInitCMAES;
			constraintViolator = checkConstraint(zNext);
			multipleDesignSitesViolator = checkMultipleDesignSites(zNext, S);
			fprintf('getNextSamplingPoint.m: constraintViolator = %d\n', constraintViolator);
			fprintf('getNextSamplingPoint.m: multipleDesignSitesViolator = %d\n', multipleDesignSitesViolator);
			zNext = reshape(zNext, 1, d); % convert to row vector
		end
	end

end

