function negativeMSE = calculateNegativeMSE(x,dmodel)
	% ---------------------------------- call DACE toolbox ---------------------------------- % 
	[y, ~, mse, ~] = predictor(x,dmodel); 
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

