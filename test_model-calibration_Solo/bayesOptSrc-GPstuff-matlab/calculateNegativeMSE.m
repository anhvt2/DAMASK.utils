function negativeSigma2 = calculateNegativeMSE(z, gp_fic, S, Y, zLB, zUB)

	z = reshape(z, 1, length(z));

	% ---------------------------------- call DACE toolbox ---------------------------------- % 
	% [y, ~, mse, ~] = predictor(x,dmodel); 
	[y, sigma2] = gp_pred(gp_fic, S, Y, z);
	sigma2 = sigma2 + gp_fic.lik.sigma2;
	negativeSigma2 = - sqrt(sigma2);
	fprintf('z:\n')
	for i = 1:length(z)
		fprintf('%.4f, ',z(i));
	end
	fprintf('\n');
	fprintf('calculateNegativeMSE.m: negativeSigma2 = %.8f\n', negativeSigma2);
	fprintf('\n');
	return 
end

