function negativeMSE = calculateNegativeMSE(x,dmodel)
	% ---------------------------------- call ooDACE toolbox ---------------------------------- % 
	x = reshape(x, 1, length(x));
	[y, rmse] = dmodel.predict(x); 
	if rmse < 0; rmse = eps; end;
	negativeMSE = - sqrt(rmse);
	fprintf('x:\n')
	for i = 1:length(x)
		fprintf('%.4f, ',x(i));
	end
	fprintf('\n');
	fprintf('calculateNegativeMSE.m: negativeMSE = %.8f\n', negativeMSE);
	fprintf('\n');
	return 
end

