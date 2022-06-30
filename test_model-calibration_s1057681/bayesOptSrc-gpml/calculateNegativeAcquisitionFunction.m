function a = calculateNegativeAcquisitionFunction(x, hyp, meanfunc, covfunc, likfunc, hypClf, meanfuncClf, covfuncClf, likfuncClf, acquisitionFunction, xLB, xUB, S, Y, F)

	% pmf of feasibility
	% [feasProb, ~, mseFeasProb, ~] = predictor(x, hypClf);
	x = reshape(x, 1, length(x)); % format to row vector
	F = 2 * F - 1; % scale from [0, 1] to [-1, 1] -- compatible with gpml
	
	try
		[feasProb, mseFeasProb] = gp(hypClf, @infEP, meanfuncClf, covfuncClf, likfuncClf, S, F, x);
		feasProb = 0.5 * (feasProb + 1); % inverse affine transformation
	catch
		feasProb = 1; % a safety guard against unconstrained problem, which might upset hypClf
	end


	% safeguard
	if feasProb > 1
		feasProb = 1;
	elseif feasProb < 0
		feasProb = 0;
	end

	if checkConstraint(x) == 0
		a = 0;
		return;
	else
		% default: GP-UCB option-2

		% [y, ~, rmse, ~] = predictor(x,hyp);
		[y, rmse] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S, Y, x);
		if rmse < 0; rmse = 0; end
		[fBest,iBest] = max(Y);

		if strcmp(acquisitionFunction, 'PI')
			%%% GP-PI 
			a = normcdf((y - fBest) / sqrt(rmse));
		elseif strcmp(acquisitionFunction, 'EI')
			%%% GP-EI
			gammaX = (y - fBest) / sqrt(rmse);
			a = sqrt(rmse) * (gammaX * normcdf(gammaX) + normpdf(gammaX) );
		elseif strcmp(acquisitionFunction, 'UCB')
			%%% GP-UCB
			delta = 0.95; % probability; see Srinivas et al. GP-UCB
			[t, d] = size(S);
			%% option-1: see Srinivas et al., theorem 1
			% volD = prod(xUB - xLB);
			% kappa = 2 * log(volD * t^2 * pi^2 / 6 / delta);
			%% option-2; see Daniel et al., Active reward learning, and Srinivas et al., theorem 2
			kappa = 2 * log(t^(d/2 + 2) * pi^2 / 3 / delta);
            %% option-3: kappa = constant
            % kappa = 2;

			a = y + sqrt(kappa) * sqrt(rmse);
		else
			fprintf('calculateNegativeAcquisitionFunction.m: acquisitionFunction option is not valid.\n');
		end

		% debug
		fprintf('Acquisition %s: mu = %0.8f\n', acquisitionFunction, y);
		fprintf('Acquisition %s: sigma = %0.8f\n', acquisitionFunction, sqrt(rmse));
		if strcmp(acquisitionFunction, 'UCB')
            fprintf('Acquisition %s: kappa = %0.8f\n', acquisitionFunction, kappa);
		end
	end

	% function called: calcAcquis
	% flip sign of acquisition function for cmaes min searching (assume maximization problem)
	% a = - calcAcquis(x,hyp);
	a = - a; %
	a = a * feasProb; % conditioned on the pmf 

	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ', x(i));
	end
	fprintf('\n');
	fprintf('Negative Acquisition Function Message: a = %.8f\n', a);
	fprintf('\n');
end

