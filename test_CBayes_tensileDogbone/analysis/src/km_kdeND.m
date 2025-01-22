function Z = km_kdeND(data,sigma,X,varargin)

% TMW: Modified for n-D and weighted KDE

% KM_KDE performs kernel density estimation (KDE) on one-dimensional data
% http://en.wikipedia.org/wiki/Kernel_density_estimation
% 
% Input:	- data: input data, one-dimensional
%           - sigma: bandwidth (sometimes called "h")
%           - nsteps: optional number of abscis points. If nsteps is an
%             array, the abscis points will be taken directly from it.
%           - range_x: optional factor for abscis expansion beyond extrema
% Output:	- x: equispaced abscis points
%			- y: estimates of p(x)
% USAGE: [xx,pp] = km_kde(data,sigma,nsteps,range_x)
%
% Author: Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2010.
% Id: km_kde.m v1.1
% This file is part of the Kernel Methods Toolbox (KMBOX) for MATLAB.
% http://sourceforge.net/p/kmbox

N = size(data,1);	% number of data points
dim = size(data,2);

Z = zeros(size(X,1),1);

if nargin > 3
    wts = varargin{1};
else
    wts = ones(size(data,1),1);
end

if isempty(sigma)
    sigma = 1/1*1.06*sqrt(var(data))*size(data,1)^(-1/(4+dim));
end
% kernel density estimation
if (min(sigma) == 0)
    Z = ones(size(X,1),1);
else
    c = 1./sqrt(2*pi*sigma.^2);
    for i=1:N
        tmp = wts(i).*c(1).*exp(-(data(i,1)-X(:,1)).^2/(2*sigma(1)^2));
        for j = 2:dim
            tmp = tmp.*c(j).*exp(-(data(i,j)-X(:,j)).^2/(2*sigma(j)^2));
        end
        Z = Z+1./N.*tmp;
    end
end
