function negativeMSE = calculateNegativeMSE(x, hyp, meanfunc, covfunc, likfunc, S, Y)
	% ---------------------------------- call DACE toolbox ---------------------------------- % 
	% [y, ~, mse, ~] = predictor(x,dmodel); 
	x = reshape(x, 1, length(x)); % format to row vector
	[y, mse] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S, Y, x);
	negativeMSE = - sqrt(mse);
	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ',x(i));
	end
	fprintf('\n');
	fprintf('calculateNegativeMSE.m: negativeMSE = %.8f\n', negativeMSE);
	fprintf('\n');
	return 
end

