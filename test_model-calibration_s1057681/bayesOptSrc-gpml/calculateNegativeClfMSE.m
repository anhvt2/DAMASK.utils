function negativeMSE = calculateNegativeClfMSE(x, hypClf, meanfuncClf, covfuncClf, likfuncClf, S, Y, F)
	% ---------------------------------- call DACE toolbox ---------------------------------- % 
	% [y, ~, mse, ~] = predictor(x,dmodel); 
	x = reshape(x, 1, length(x)); % format to row vector
	% [y, mse] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S, Y, x);
	F = 2 * F - 1; % scale from [0, 1] to [-1, 1] -- compatible with gpml
	[y, mse] = gp(hypClf, @infEP, meanfuncClf, covfuncClf, likfuncClf, S, F, x);
	y = 0.5 * (y + 1); % inverse affine transformation
	negativeMSE = - sqrt(mse);
	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ',x(i));
	end
	fprintf('\n');
	fprintf('calculateNegativeClfMSE.m: negativeMSE = %.8f\n', negativeMSE);
	fprintf('\n');
	return 
end

