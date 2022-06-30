function Phi = normcdf(x)
	% return normal CDF
	Phi = 0.5 * (1 + erf(x/sqrt(2)));
end
