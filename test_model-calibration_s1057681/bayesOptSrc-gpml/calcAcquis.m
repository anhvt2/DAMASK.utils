function a = calcAcquis(x,dmodel)
	% combine with calcNegAcquis.m
	if checkConstraint(x) == 0
		a = 0;
		return;
	else
		[y, ~, mse, ~] = predictor(x,dmodel);
		%% GP-PI 



		%% GP-EI
		[fBest,iBest] = min(dmodel.origY);
		gammaX = ( y - fBest )/sqrt(mse);
		a = sqrt(mse) * (gammaX * normcdf(gammaX) + normpdf(gammaX) );

		%% GP-UCB
		% kappa = 1e2;
		% a = y + kappa * sqrt(mse);
		fprintf('Acquisition: mu = %0.8f\n', y);
		fprintf('Acquisition: mse = %0.8f\n', mse);
		return;
	end

end

