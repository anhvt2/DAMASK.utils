function a = calcNegAcquis(x,dmodel)

	if checkConstraint(x) == 0
		a = 0;
		return;
	else
		% default: GP-UCB option-2

		[y, ~, rmse, ~] = predictor(x,dmodel);
		[fBest,iBest] = min(dmodel.origY);
		%%% GP-PI 
		% a = normcdf((y - fBest) / rmse);

		%%% GP-EI
		% gammaX = ( y - fBest )/sqrt(rmse);
		% a = sqrt(rmse) * (gammaX * normcdf(gammaX) + normpdf(gammaX) );

		%%% GP-UCB
		delta = 0.95; % probability; see Srinivas et al. GP-UCB
		[t, d] = size(dmodel.S);
		%% option-1: see Srinivas et al., theorem 1
		% volD = prod(dmodel.xUB - dmodel.xLB);
		% kappa = 2 * log(volD * t^2 * pi^2 / 6 / delta);
		%% option-2; see Daniel et al., Active reward learning, and Srinivas et al., theorem 2
		kappa = 2 * log(t^(d/2 + 2) * pi^2 / 3 / delta);

		a = y + sqrt(kappa) * sqrt(rmse);

		fprintf('Acquisition: mu = %0.8f\n', y);
		fprintf('Acquisition: sigma = %0.8f\n', rmse);
		return;
	end

	% function called: calcAcquis
	% flip sign of acquisition function 
	% a = - calcAcquis(x,dmodel);
	a = -a; %

	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ', x(i));
	end
	fprintf('\n');
	fprintf('Negative Acquisition Function Message: a = %.8f\n', a);
	fprintf('\n');
end

