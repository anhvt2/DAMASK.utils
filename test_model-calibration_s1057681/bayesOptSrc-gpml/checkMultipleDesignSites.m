function a = checkMultipleDesignSites(xNext, S)
	% return
	% 	1 if xNext is duplicated in S
	%	0 if xNext is NOT duplicated in S
	[m, dp] = size(S);
	d = length(xNext);
	xNext = reshape(xNext, 1, d);
	if dp ~= d; S = S'; end
	tol = 1e-4; % 1e-9 is too small; this 'tol' parameter assumes xScale has already been applied
	for i = 1:m
		if norm(S(i, :) - xNext) < tol
			a = 1; 
			sprintf('checkMultipleDesignSites.m: Duplicates found.\n');
			sprintf('checkMultipleDesignSites.m: Retry searching for the next sampling point.\n');
			return;
		end
	end
	a = 0;
end
