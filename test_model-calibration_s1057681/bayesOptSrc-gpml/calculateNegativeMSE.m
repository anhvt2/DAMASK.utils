function negativeMSE = calculateNegativeMSE(x, hyp, meanfunc, covfunc, likfunc, S, Y)
	% ---------------------------------- call DACE toolbox ---------------------------------- % 
	% [y, ~, s2, ~] = predictor(x,dmodel); 
	x = reshape(x, 1, length(x)); % format to row vector
	[y, s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S, Y, x);
	negativeMSE = - sqrt(s2);
	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ',x(i));
	end
	fprintf('\n');
	fprintf('calculateNegativeMSE.m: negativeMSE = %.4e\n', negativeMSE);
	fprintf('\n');
	return 
end

