function lik = lik_loggaussian(varargin)
%LIK_LOGGAUSSIAN  Create a right censored log-Gaussian likelihood structure 
%
%  Description
%    LIK = LIK_LOGGAUSSIAN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a likelihood structure for right censored log-Gaussian
%    survival model in which the named parameters have the specified
%    values. Any unspecified parameters are set to default values.
%  
%    LIK = LIK_LOGGAUSSIAN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a likelihood structure with the named parameters
%    altered with the specified values.
%
%    Parameters for log-Gaussian likelihood [default]
%      sigma2       - variance [1]
%      sigma2_prior - prior for sigma2 [prior_logunif]
%  
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    The likelihood is defined as follows:
%                  __ n
%      p(y|f, z) = || i=1 [ (2*pi*s^2)^(-(1-z_i)/2)*y_i^-(1-z_i)
%                            *exp(-1/(2*s^2)*(1-z_i)*(log(y_i) - f_i)^2)
%                            *(1-norm_cdf((log(y_i)-f_i)/s))^z_i ]
%                           
%
%    where s is the standard deviation of loggaussian distribution.
%    z is a vector of censoring indicators with z = 0 for uncensored event
%    and z = 1 for right censored event. 
%
%    When using the log-Gaussian likelihood you can give the
%    vector z as an extra parameter to each function that requires
%    also y. For example, you can call gp_optim as follows:
%      gp_optim(gp, x, y, 'z', z)
%    If z is not given or it is empty, then usual likelihood for
%    uncensored data is used
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
% Copyright (c) 2012 Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_LOGGAUSSIAN';
  ip=iparser(ip,'addOptional','lik', [], @isstruct);
  ip=iparser(ip,'addParamValue','sigma2',1, @(x) isscalar(x) && x>0);
  ip=iparser(ip,'addParamValue','sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip=iparser(ip,'parse',varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.type = 'Log-Gaussian';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Log-Gaussian')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_loggaussian_pak;
    lik.fh.unpak = @lik_loggaussian_unpak;
    lik.fh.lp = @lik_loggaussian_lp;
    lik.fh.lpg = @lik_loggaussian_lpg;
    lik.fh.ll = @lik_loggaussian_ll;
    lik.fh.llg = @lik_loggaussian_llg;    
    lik.fh.llg2 = @lik_loggaussian_llg2;
    lik.fh.llg3 = @lik_loggaussian_llg3;
    lik.fh.tiltedMoments = @lik_loggaussian_tiltedMoments;
    lik.fh.siteDeriv = @lik_loggaussian_siteDeriv;
    lik.fh.invlink = @lik_loggaussian_invlink;
    lik.fh.predy = @lik_loggaussian_predy;
    lik.fh.recappend = @lik_loggaussian_recappend;
    lik.fh.predcdf = @lik_loggaussian_predcdf;
  end

end

function [w,s,h] = lik_loggaussian_pak(lik)
%LIK_LOGGAUSSIAN_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_LOGGAUSSIAN_PAK(LIK) takes a likelihood structure LIK and
%    combines the parameters into a single row vector W. This is a 
%    mandatory subfunction used for example in energy and gradient 
%    computations.
%     
%       w = log(lik.sigma2)
%
%   See also
%   LIK_LOGGAUSSIAN_UNPAK, GP_PAK
  
  w=[];s={};h=[];
  if ~isempty(lik.p.sigma2)
    w = log(lik.sigma2);
    s = [s; 'log(loggaussian.sigma2)'];
    h = [h 0];
    [wh,sh, hh] = lik.p.sigma2.fh.pak(lik.p.sigma2);
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end
end


function [lik, w] = lik_loggaussian_unpak(lik, w)
%LIK_LOGGAUSSIAN_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    [LIK, W] = LIK_LOGGAUSSIAN_UNPAK(W, LIK) takes a likelihood
%    structure LIK and extracts the parameters from the vector W
%    to the LIK structure. This is a mandatory subfunction used 
%    for example in energy and gradient computations.
%     
%   Assignment is inverse of  
%       w = log(lik.sigma2)
%
%   See also
%   LIK_LOGGAUSSIAN_PAK, GP_UNPAK

  if ~isempty(lik.p.sigma2)
    lik.sigma2 = exp(w(1));
    w = w(2:end);
    [p, w] = lik.p.sigma2.fh.unpak(lik.p.sigma2, w);
    lik.p.sigma2 = p;
  end
end


function lp = lik_loggaussian_lp(lik, varargin)
%LIK_LOGGAUSSIAN_LP  log(prior) of the likelihood parameters
%
%  Description
%    LP = LIK_LOGGAUSSIAN_LP(LIK) takes a likelihood structure LIK and
%    returns log(p(th)), where th collects the parameters. This subfunction 
%    is needed when there are likelihood parameters.
%
%  See also
%    LIK_LOGGAUSSIAN_LLG, LIK_LOGGAUSSIAN_LLG3, LIK_LOGGAUSSIAN_LLG2, GPLA_E
  

% If prior for sigma2 parameter, add its contribution
  lp=0;
  if ~isempty(lik.p.sigma2)
    lp = lik.p.sigma2.fh.lp(lik.sigma2, lik.p.sigma2) +log(lik.sigma2);
  end
  
end


function lpg = lik_loggaussian_lpg(lik)
%LIK_LOGGAUSSIAN_LPG  d log(prior)/dth of the likelihood 
%                parameters th
%
%  Description
%    E = LIK_LOGGAUSSIAN_LPG(LIK) takes a likelihood structure LIK and
%    returns d log(p(th))/dth, where th collects the parameters. This 
%    subfunction is needed when there are likelihood parameters.
%
%  See also
%    LIK_LOGGAUSSIAN_LLG, LIK_LOGGAUSSIAN_LLG3, LIK_LOGGAUSSIAN_LLG2, GPLA_G
  
  lpg=[];
  if ~isempty(lik.p.sigma2)            
    % Evaluate the gprior with respect to sigma2
    ggs = lik.p.sigma2.fh.lpg(lik.sigma2, lik.p.sigma2);
    lpg = ggs(1).*lik.sigma2 + 1;
    if length(ggs) > 1
      lpg = [lpg ggs(2:end)];
    end
  end
end  

function ll = lik_loggaussian_ll(lik, y, f, z)
%LIK_LOGGAUSSIAN_LL  Log likelihood
%
%  Description
%    LL = LIK_LOGGAUSSIAN_LL(LIK, Y, F, Z) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z, and
%    latent values F. Returns the log likelihood, log p(y|f,z). 
%    This subfunction is needed when using Laplace approximation 
%    or MCMC for inference with non-Gaussian likelihoods. This 
%    subfunction is also used in information criteria (DIC, WAIC) 
%    computations.
%
%  See also
%    LIK_LOGGAUSSIAN_LLG, LIK_LOGGAUSSIAN_LLG3, LIK_LOGGAUSSIAN_LLG2, GPLA_E
  
  if numel(z)==0
    z=0;
  end

  s2 = lik.sigma2;
  if sum(z)>1
    z=logical(z);
    ll=zeros(size(f));
    ll(~z) = -1/2*log(2*pi*s2) - log(y(~z)) - 1./(2*s2).*(log(y(~z))-f(~z)).^2;
    ll(z) = log(1-norm_cdf((log(y(z))-f(z))./sqrt(s2)));
    ll=sum(ll);
  else
    ll = sum(-1/2*log(2*pi*s2) - log(y) - 1./(2*s2).*(log(y)-f).^2);
  end

end

function llg = lik_loggaussian_llg(lik, y, f, param, z)
%LIK_LOGGAUSSIAN_LLG  Gradient of the log likelihood
%
%  Description 
%    LLG = LIK_LOGGAUSSIAN_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z and
%    latent values F. Returns the gradient of the log likelihood
%    with respect to PARAM. At the moment PARAM can be 'param' or
%    'latent'. This subfunction is needed when using Laplace 
%    approximation or MCMC for inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LOGGAUSSIAN_LL, LIK_LOGGAUSSIAN_LLG2, LIK_LOGGAUSSIAN_LLG3, GPLA_E

  if numel(z)==0
    z=0;
  end

  s2 = lik.sigma2;
  switch param
    case 'param'      
      if sum(z)>1
        z=logical(z);
        llg=zeros(size(f));
        r = log(y(~z))-f(~z);
        llg(~z) = -1./(2.*s2) + r.^2./(2.*s2^2);
        r = log(y(z))-f(z);
        llg(z) = 1./(1-norm_cdf(r/sqrt(s2))).*(r./(sqrt(2.*pi).*2.*s2.^(3/2)).*exp(-1/(2.*s2).*r.^2));
        llg=sum(llg);
      else
        r = log(y)-f;
        llg = sum(-1./(2.*s2) + r.^2./(2.*s2^2));
      end
      % correction for the log transformation
      llg = llg.*s2;
    case 'latent'
      if sum(z)>1
        z=logical(z);
        llg=zeros(size(f));
        r = log(y(~z))-f(~z);
        llg(~z) = 1./s2.*r;
        r = log(y(z))-f(z);
        llg(z) = 1./(1-norm_cdf(r/sqrt(s2))).*(1/sqrt(2*pi*s2) .* exp(-1/(2.*s2).*r.^2));
      else
        r = log(y)-f;
        llg = 1./s2.*r;
      end
  end
end

function llg2 = lik_loggaussian_llg2(lik, y, f, param, z)
%LIK_LOGGAUSSIAN_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_LOGGAUSSIAN_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z, and
%    latent values F. Returns the hessian of the log likelihood
%    with respect to PARAM. At the moment PARAM can be only
%    'latent'. LLG2 is a vector with diagonal elements of the
%    Hessian matrix (off diagonals are zero). This subfunction 
%    is needed when using Laplace approximation or EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LOGGAUSSIAN_LL, LIK_LOGGAUSSIAN_LLG, LIK_LOGGAUSSIAN_LLG3, GPLA_E

  if numel(z)==0
    z=0;
  end

  s2 = lik.sigma2;
  switch param
    case 'param'
      
    case 'latent'
      if sum(z)>1
        z=logical(z);
        llg2=zeros(size(f));
        llg2(~z) = -1./s2;
        r = log(y(z))-f(z);
        a=(1-norm_cdf(r/sqrt(s2)));
        llg2(z) = (-exp(-r.^2/s2)./(2*pi*s2.*a.^2) ...
                + r./(sqrt(2*pi).*s2^(3/2).*a).*exp(-r.^2./(2*s2)));
      else
        llg2 = repmat(-1./s2,size(f));
      end
    case 'latent+param'
      if sum(z)>1
        z=logical(z);
        llg2=zeros(size(f));
        r = log(y(~z))-f(~z);
        llg2(~z) = -1./s2^2.*r;
        r = log(y(z))-f(z);
        a=(1-norm_cdf(r/sqrt(s2)));
        llg2(z) = (-r./(4*pi*s2^2.*a.^2) ...
                .* exp(-r.^2./s2) + (-1 + r.^2/s2)./a.*1./(sqrt(2*pi)*2*s2^(3/2)).*exp(-r.^2./(2*s2)));
      else
        r = log(y)-f;
        llg2 = -1./s2^2.*r;
      end
      % correction due to the log transformation
      llg2 = llg2.*s2;
  end
end    

function llg3 = lik_loggaussian_llg3(lik, y, f, param, z)
%LIK_LOGGAUSSIAN_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_LOGGAUSSIAN_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, survival times Y, censoring indicators Z and
%    latent values F and returns the third gradients of the log
%    likelihood with respect to PARAM. At the moment PARAM can be
%    only 'latent'. LLG3 is a vector with third gradients. This 
%    subfunction is needed when using Laplace approximation for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    LIK_LOGGAUSSIAN_LL, LIK_LOGGAUSSIAN_LLG, LIK_LOGGAUSSIAN_LLG2, GPLA_E, GPLA_G

  if numel(z)==0
    z=0;
  end

  s2 = lik.sigma2;
  switch param
    case 'param'
      
    case 'latent'
      if sum(z)>1
        z=logical(z);
        llg3=zeros(size(f));
        r = log(y(z)) - f(z);
        a=(1-norm_cdf(r/sqrt(s2)));
        llg3(z) = 2./a.^3.*1./(2*pi*s2)^(3/2).*exp(-3/(2*s2)*r.^2) ...
               - 1./a.^2.*r./(pi*s2^2).*exp(-r.^2./s2) ...
               - 1./a.^2.*r./(2*pi*s2^2).*exp(-r.^2/s2) ...
               - 1./a.^1.*1./(s2^(3/2)*sqrt(2*pi)).*exp(-r.^2/(2*s2)) ...
               + 1./a.^1.*r.^2./(sqrt(2*pi*s2)*s2^2).*exp(-r.^2/(2*s2));
      else
        llg3=zeros(size(f));
      end
    case 'latent2+param'
      if sum(z)>1
        z=logical(z);
        llg3=zeros(size(f));
        llg3(~z)=1./s2^2;
        r = log(y(z)) - f(z);
        a=(1-norm_cdf(r./sqrt(s2)));
        llg3(z) = (1./a.^3.*r./(sqrt(8*pi^3).*s2.^(5/2)).*exp(-3/(2.*s2).*r.^2) ...
                   + 1./a.^2.*1./(4.*pi.*s2^2).*exp(-r.^2./s2) ...
                   - 1./a.^2.*r.^2./(2*pi*s2^3).*exp(-r.^2./s2) ...
                   + 1./a.^2.*1./(4*pi*s2^2).*exp(-r.^2/s2) ...
                   - 1./a.^1.*r./(sqrt(2*pi)*2*s2^(5/2)).*exp(-r.^2/(2*s2)) ...
                   - 1./a.^2.*r.^2./(4*pi*s2^3).*exp(-r.^2/s2) ...
                   - 1./a.^1.*r./(sqrt(2*pi)*s2^(5/2)).*exp(-r.^2/(2*s2)) ...
                   + 1./a.^1.*r.^3./(sqrt(2*pi)*2*s2^(7/2)).*exp(-r.^2/(2*s2)));
      else
        llg3 = repmat(1./s2^2,size(f));
      end
      % correction due to the log transformation
      llg3 = llg3.*s2;
  end
end

function [logM_0, m_1, sigm2hati1] = lik_loggaussian_tiltedMoments(lik, y, i1, sigm2_i, myy_i, z)
%LIK_LOGGAUSSIAN_TILTEDMOMENTS  Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_LOGGAUSSIAN_TILTEDMOMENTS(LIK, Y, I, S2,
%    MYY, Z) takes a likelihood structure LIK, survival times
%    Y, censoring indicators Z, index I and cavity variance S2 and
%    mean MYY. Returns the zeroth moment M_0, mean M_1 and
%    variance M_2 of the posterior marginal (see Rasmussen and
%    Williams (2006): Gaussian processes for Machine Learning,
%    page 55).  This subfunction is needed when using EP for 
%    inference with non-Gaussian likelihoods.
%
%  See also
%    GPEP_E
  
 if numel(z)==0
   z=zeros(size(y));
 end
  
  yy = y(i1);
  yc = 1-z(i1);
  s2 = lik.sigma2;
  logM_0=zeros(size(yy));
  m_1=zeros(size(yy));
  sigm2hati1=zeros(size(yy));  
  
  ind1=yc==1;
  ind2=yc==0;
  
  if ~isscalar(sigm2_i)
    if sum(ind1)>0
      logM_0(ind1)=-log(yy(ind1))+norm_lpdf(myy_i(ind1), log(yy(ind1)), sqrt(s2+sigm2_i(ind1)));
      sigm2hati1(ind1)=1./(1./s2+1./sigm2_i(ind1));
      m_1(ind1)=sigm2hati1(ind1).*(myy_i(ind1)./sigm2_i(ind1) + log(yy(ind1))/s2);
    end
    if sum(ind2)>0
      zi=(myy_i(ind2)-log(yy(ind2)))./sqrt(s2+sigm2_i(ind2));
      logM_0(ind2)=log(norm_cdf(zi));
      m_1(ind2)=myy_i(ind2)+sigm2_i(ind2).*norm_pdf(zi)./(norm_cdf(zi).*sqrt(s2+sigm2_i(ind2)));
      m_2=2*myy_i(ind2).*m_1(ind2) - myy_i(ind2).^2 + sigm2_i(ind2) - ...
        sigm2_i(ind2).^2.*norm_pdf(zi).*zi./(norm_cdf(zi).*(s2+sigm2_i(ind2)));
      sigm2hati1(ind2) = m_2 - m_1(ind2).^2;        
    end
    if any(sigm2hati1 >= sigm2_i)
      sigm2hati1(sigm2hati1 >= sigm2_i)=sigm2_i(sigm2hati1 >= sigm2_i)-eps;
      %error('lik_loggaussian_tilted_moments: sigm2hati1 >= sigm2_i');
    end
  else
    if sum(ind1)>0
      logM_0(ind1)=-log(yy(ind1))+norm_lpdf(myy_i(ind1), log(yy(ind1)), sqrt(s2+sigm2_i));
      sigm2hati1(ind1)=1./(1./s2+1./sigm2_i);
      m_1(ind1)=sigm2hati1(ind1).*(myy_i(ind1)./sigm2_i + log(yy(ind1))/s2);
    end
    if sum(ind2)>0
      zi=(myy_i(ind2)-log(yy(ind2)))./sqrt(s2+sigm2_i);
      logM_0(ind2)=log(norm_cdf(zi));
      m_1(ind2)=myy_i(ind2)+sigm2_i.*norm_pdf(zi)./(norm_cdf(zi).*sqrt(s2+sigm2_i));
      m_2=2*myy_i(ind2).*m_1(ind2) - myy_i(ind2).^2 + sigm2_i - ...
        sigm2_i.^2.*norm_pdf(zi).*zi./(norm_cdf(zi).*(s2+sigm2_i));
      sigm2hati1(ind2) = m_2 - m_1(ind2).^2;        
    end
    if any(sigm2hati1 >= sigm2_i)
      sigm2hati1(sigm2hati1 >= sigm2_i)=sigm2_i-eps;
      %error('lik_loggaussian_tilted_moments: sigm2hati1 >= sigm2_i');
    end
  end
  
%   sigm2hati12=sigm2hati1;
%   m_12=m_1;
%   logM_02=logM_0;
  
%   for i=1:length(i1)
%     % get a function handle of an unnormalized tilted distribution
%     % (likelihood * cavity = Negative-binomial * Gaussian)
%     % and useful integration limits
%     [tf,minf,maxf]=init_loggaussian_norm(yy(i),myy_i(i),sigm2_i(i),yc(i),s2);
%     
%     % Integrate with quadrature
%     RTOL = 1.e-6;
%     ATOL = 1.e-10;
%     [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
% %     ll(~z) = -1/2*log(2*pi*s2) - log(y(~z)) - 1./(2*s2).*(log(y(~z))-f(~z)).^2;
% %     ll(z) = log(1-norm_cdf((log(y(z))-f(z))./sqrt(s2)));
%     if isnan(m_0)
%       logM_0=NaN;
%       return
%     end
%     sigm2hati1(i) = m_2 - m_1(i).^2;
%     
%     % If the second central moment is less than cavity variance
%     % integrate more precisely. Theoretically for log-concave
%     % likelihood should be sigm2hati1 < sigm2_i.
%     if sigm2hati1(i) >= sigm2_i(i)
%       ATOL = ATOL.^2;
%       RTOL = RTOL.^2;
%       [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
%       sigm2hati1(i) = m_2 - m_1(i).^2;
%       if sigm2hati1(i) >= sigm2_i(i)
%          sigm2hati1(i)=sigm2_i(i)-eps;
%          %error('lik_loggaussian_tilted_moments: sigm2hati1 >= sigm2_i');
%       end
%     end
%     logM_0(i) = log(m_0);
%   end
%   1;
end

function [g_i] = lik_loggaussian_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)
%LIK_LOGGAUSSIAN_SITEDERIV  Evaluate the expectation of the gradient
%                      of the log likelihood term with respect
%                      to the likelihood parameters for EP 
%
%  Description [M_0, M_1, M2] =
%    LIK_LOGGAUSSIAN_SITEDERIV(LIK, Y, I, S2, MYY, Z) takes a
%    likelihood structure LIK, survival times Y, expected
%    counts Z, index I and cavity variance S2 and mean MYY. 
%    Returns E_f [d log p(y_i|f_i) /d a], where a is the
%    likelihood parameter and the expectation is over the
%    marginal posterior. This term is needed when evaluating the
%    gradients of the marginal likelihood estimate Z_EP with
%    respect to the likelihood parameters (see Seeger (2008):
%    Expectation propagation for exponential families). This 
%    subfunction is needed when using EP for inference with 
%    non-Gaussian likelihoods and there are likelihood parameters.
%
%  See also
%    GPEP_G

  if numel(z)==0
    z=zeros(size(y));
  end

  yy = y(i1);
  yc = 1-z(i1);
  s2 = lik.sigma2;
  
  % get a function handle of an unnormalized tilted distribution 
  % (likelihood * cavity = Log-Gaussian * Gaussian)
  % and useful integration limits
  [tf,minf,maxf]=init_loggaussian_norm(yy,myy_i,sigm2_i,yc,s2);
  % additionally get function handle for the derivative
  td = @deriv;
  
  % Integrate with quadgk
  if yc==1
    m_0=1/yy.*norm_pdf(myy_i, log(yy), sqrt(s2+sigm2_i));
  else
    zi=(myy_i-log(yy))./sqrt(s2+sigm2_i);
    m_0=norm_cdf(zi);
%     [m_0, fhncnt] = quadgk(tf, minf, maxf);
  end
  [g_i, fhncnt] = quadgk(@(f) td(f, yy, yc, s2).*tf(f)./m_0, minf, maxf);
  g_i = g_i.*s2;
end

  function g = deriv(f, yy, yc, s2)
    if yc==0
      r=log(yy)-f;
      g = 1./(1-norm_cdf(r/sqrt(s2))) ...
           .* (r./(sqrt(2.*pi).*2.*s2.^(3/2)).*exp(-1/(2.*s2).*r.^2));
    else
      r=log(yy)-f;
      g = -1./(2.*s2) + r.^2./(2.*s2^2);
    end
  end

function [lpy, Ey, Vary] = lik_loggaussian_predy(lik, Ef, Varf, yt, zt)
%LIK_LOGGAUSSIAN_PREDY  Returns the predictive mean, variance and density of y
%
%  Description   
%    LPY = LIK_LOGGAUSSIAN_PREDY(LIK, EF, VARF YT, ZT)
%    Returns logarithm of the predictive density PY of YT, that is 
%        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
%    This requires also the survival times YT, censoring indicators ZT.
%    This subfunction is needed when computing posterior predictive 
%    distributions for future observations.
%
%    [LPY, EY, VARY] = LIK_LOGGAUSSIAN_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction 
%    is needed when computing posterior predictive distributions for 
%    future observations.
%        
%
%  See also
%    GPLA_PRED, GPEP_PRED, GPMC_PRED

  if numel(zt)==0
    zt=zeros(size(yt));
  end

  yc = 1-zt;
  s2 = lik.sigma2;
  
  Ey=[];
  Vary=[];

  % Evaluate the posterior predictive densities of the given observations
  lpy = zeros(length(yt),1);
  if (size(Ef,2) > 1) && (size(Ef,2) > 1) && size(yt,2) == 1
    % Approximate integral with sum of grid points when using corrected
    % marginal posterior pf
    for i1=1:length(yt)
      py = arrayfun(@(f) exp(lik.fh.ll(lik, yt(i1), f, zt(i1))), Ef(i1,:));
      pf = Varf(i1,:)./sum(Varf(i1,:));
      lpy(i1) = log(sum(py.*pf));
    end
  else
%     for i1=1:length(yt)
%       if abs(Ef(i1))>700
%         lpy(i1) = NaN;
%       else
%         % get a function handle of the likelihood times posterior
%         % (likelihood * posterior = Negative-binomial * Gaussian)
%         % and useful integration limits
%         [pdf,minf,maxf]=init_loggaussian_norm(...
%           yt(i1),Ef(i1),Varf(i1),yc(i1),s2);
%         % integrate over the f to get posterior predictive distribution
%         lpy(i1) = log(quadgk(pdf, minf, maxf));
%       end
%       
%     end
    ind1=yc==1;
    ind2=yc==0;
    
    if sum(ind1)>0
      if ~isempty(yt)
        lpy(ind1,:)=-log(yt(ind1))+norm_lpdf(Ef(ind1), log(yt(ind1)), sqrt(s2+Varf(ind1)));
      end
      Ey(ind1,:) = exp(Ef(ind1)+Varf(ind1)/2);
      Vary(ind1,:) = (exp(Varf(ind1))-1).*exp(2*Ef(ind1)+Varf(ind1));
    end
    if sum(ind2)>0
      if ~isempty(yt)
        zi=(Ef(ind2)-log(yt(ind2)))./sqrt(s2+Varf(ind2));
        lpy(ind2,:)=log(norm_cdf(zi));
      end
      py1 = norm_cdf(Ef(ind2)./sqrt(1+Varf(ind2)));
      Ey(ind2,:) = 2*py1 - 1;      
      Vary(ind2,:) = 1-Ey(ind2).^2;
    end
  end
end

function [df,minf,maxf] = init_loggaussian_norm(yy,myy_i,sigm2_i,yc,s2)
%INIT_LOGGAUSSIAN_NORM
%
%  Description
%    Return function handle to a function evaluating
%    loggaussian * Gaussian which is used for evaluating
%    (likelihood * cavity) or (likelihood * posterior) Return
%    also useful limits for integration. This is private function
%    for lik_loggaussian. This subfunction is needed by subfunctions
%    tiltedMoments, siteDeriv and predy.
%  
%  See also
%    LIK_LOGGAUSSIAN_TILTEDMOMENTS, LIK_LOGGAUSSIAN_SITEDERIV,
%    LIK_LOGGAUSSIAN_PREDY
  
% avoid repetitive evaluation of constant part
  ldconst = -yc./2.*log(2*pi*s2) -yc.*log(yy) ...
            - log(sigm2_i)/2 - log(2*pi)/2;
  
  % Create function handle for the function to be integrated
  df = @(f) loggaussian_norm(f, ldconst, yy, yc, s2, myy_i, sigm2_i);
  % use log to avoid underflow, and derivates for faster search
  ld = @log_loggaussian_norm;
  ldg = @log_loggaussian_norm_g;
  ldg2 = @log_loggaussian_norm_g2;

  % Set the limits for integration
  if yc==0
    % with yc==0, the mode of the likelihood is not defined
    if myy_i>log(yy)
      % the log likelihood is flat on this side
      % use the mode of the Gaussian (cavity or posterior)
      modef = myy_i;
    else
      % the log likelihood is approximately f on this side
      modef = min(myy_i+sigm2_i,log(yy)+sqrt(s2));
    end
  else
    % use precision weighted mean of the Gaussian approximation
    % of the loggaussian likelihood and Gaussian
    mu=log(yy);
    modef = (myy_i/sigm2_i + mu/s2)/(1/sigm2_i + 1/s2);
  end
  modef0=modef;
  % find the mode of the integrand using Newton iterations
  % few iterations is enough, since the first guess in the right direction
  niter=4;       % number of Newton iterations
  mindelta=1e-6; % tolerance in stopping Newton iterations
  for ni=1:niter
    g=ldg(modef, ldconst, yy, yc, s2, myy_i, sigm2_i);
    h=ldg2(modef, ldconst, yy, yc, s2, myy_i, sigm2_i);
    delta=-g/h;
    modef=modef+delta;
    if abs(delta)<mindelta||isnan(modef)
      break
    end
  end
  if isnan(modef)
    modef=modef0;
    modes=sqrt(s2);
  else
    modes=sqrt(-1/h);
  end
  % integrand limits based on Gaussian approximation at mode
  minf=modef-8*modes;
  maxf=modef+8*modes;
  modeld=ld(modef, ldconst, yy, yc, s2, myy_i, sigm2_i);
  iter=0;
  % check that density at end points is low enough
  lddiff=20; % min difference in log-density between mode and end-points
  minld=ld(minf, ldconst, yy, yc, s2, myy_i, sigm2_i);
  step=1;
  while minld>(modeld-lddiff)
    minf=minf-step*modes;
    minld=ld(minf, ldconst, yy, yc, s2, myy_i, sigm2_i);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_loggaussian -> init_loggaussian_norm: ' ...
             'integration interval minimun not found ' ...
             'even after looking hard!'])
    end
  end
  maxld=ld(maxf, ldconst, yy, yc, s2, myy_i, sigm2_i);
  iter=0;
  step=1;
  while maxld>(modeld-lddiff)
    maxf=maxf+step*modes;
    maxld=ld(maxf, ldconst, yy, yc, s2, myy_i, sigm2_i);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_loggaussian -> init_loggaussian_norm: ' ...
             'integration interval maximun not found ' ...
             'even after looking hard!'])
    end
  end
  
end
  function integrand = loggaussian_norm(f, ldconst, yy, yc, s2, myy_i, sigm2_i)
  % loggaussian * Gaussian
    if yc
      % observed
      lik = -1./(2*s2).*(log(yy)-f).^2;
    else
      % censored
      lik = log(1-norm_cdf((log(yy)-f)/sqrt(s2)));
    end
    integrand = exp(ldconst ...
                    + lik ...
                    -0.5*(f-myy_i).^2./sigm2_i);
  end

  function log_int = log_loggaussian_norm(f, ldconst, yy, yc, s2, myy_i, sigm2_i)
  % log(loggaussian * Gaussian)
  % log_loggaussian_norm is used to avoid underflow when searching
  % integration interval
    if yc
      % observed
      lik = -1./(2*s2).*(log(yy)-f).^2;
    else
      % censored
      lik = log(1-norm_cdf((log(yy)-f)/sqrt(s2)));
    end
    log_int = ldconst ...
              + lik ...
              -0.5*(f-myy_i).^2./sigm2_i;
  end

  function g = log_loggaussian_norm_g(f, ldconst, yy, yc, s2, myy_i, sigm2_i)
  % d/df log(loggaussian * Gaussian)
  % derivative of log_loggaussian_norm
    if yc
      % observed
      glik = 1./s2.*(log(yy)-f);
    else
      % censored
      glik = 1./(1-norm_cdf((log(yy)-f)/sqrt(s2))).*1/sqrt(2*pi*s2)*exp(-(log(yy)-f).^2./(2*s2));
    end
    g = + glik  ...
        + (myy_i - f)./sigm2_i;
  end

  function g2 = log_loggaussian_norm_g2(f, ldconst, yy, yc, s2, myy_i, sigm2_i)
  % d^2/df^2 log(loggaussian * Gaussian)
  % second derivate of log_loggaussian_norm
    if yc
      % observed
      g2lik = -1./s2;
    else
      % censored
      g2lik = 1./(1-norm_cdf((log(yy)-f)/sqrt(s2))).*1/sqrt(2*pi*s2)*exp(-(log(yy)-f).^2./(2*s2));
    end
    g2 =  + (1-yc).*(-exp(-(log(yy)-f).^2/s2)./(2*pi*s2.*(1-norm_cdf((log(yy)-f)/sqrt(s2))).^2) ...
              + (log(yy)-f)./(sqrt(2*pi).*s2^(3/2).*(1-norm_cdf((log(yy)-f)/sqrt(s2)))).*exp(-(log(yy)-f).^2./(2*s2))) ...
              -1/sigm2_i;
  end


function cdf = lik_loggaussian_predcdf(lik, Ef, Varf, yt)
%LIK_LOGGAUSSIAN_PREDCDF  Returns the predictive cdf evaluated at yt 
%
%  Description   
%    CDF = LIK_LOGGAUSSIAN_PREDCDF(LIK, EF, VARF, YT)
%    Returns the predictive cdf evaluated at YT given likelihood
%    structure LIK, posterior mean EF and posterior Variance VARF
%    of the latent variable. This subfunction is needed when using
%    functions gp_predcdf or gp_kfcv_cdf.
%
%  See also
%    GP_PREDCDF
  
  s2 = lik.sigma2;
  
  % Evaluate the posterior predictive densities of the given observations
  cdf = zeros(length(yt),1);
  for i1=1:length(yt)
    % Get a function handle of the likelihood times posterior
    % (likelihood * posterior = log-Gaussian * Gaussian)
    % and useful integration limits.
    % yc=0 when evaluating predictive cdf
    [pdf,minf,maxf]=init_loggaussian_norm(...
      yt(i1),Ef(i1),Varf(i1),0,s2);
    % integrate over the f to get posterior predictive distribution
    cdf(i1) = 1-quadgk(pdf, minf, maxf);
  end
  
end

function p = lik_loggaussian_invlink(lik, f, z)
%LIK_LOGGAUSSIAN Returns values of inverse link function
%             
%  Description 
%    P = LIK_LOGGAUSSIAN_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values of inverse link function P.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_LOGGAUSSIAN_LL, LIK_LOGGAUSSIAN_PREDY

p = exp(f);
end

function reclik = lik_loggaussian_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = GPCF_LOGGAUSSIAN_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction 
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also
%    GP_MC

  if nargin == 2
    % Initialize the record
    reclik.type = 'Log-Gaussian';

    % Initialize parameter
    reclik.sigma2 = [];

    % Set the function handles
    reclik.fh.pak = @lik_loggaussian_pak;
    reclik.fh.unpak = @lik_loggaussian_unpak;
    reclik.fh.lp = @lik_loggaussian_lp;
    reclik.fh.lpg = @lik_loggaussian_lpg;
    reclik.fh.ll = @lik_loggaussian_ll;
    reclik.fh.llg = @lik_loggaussian_llg;    
    reclik.fh.llg2 = @lik_loggaussian_llg2;
    reclik.fh.llg3 = @lik_loggaussian_llg3;
    reclik.fh.tiltedMoments = @lik_loggaussian_tiltedMoments;
    reclik.fh.invlink = @lik_loggaussian_invlink;
    reclik.fh.predy = @lik_loggaussian_predy;
    reclik.fh.recappend = @lik_loggaussian_recappend;
    reclik.p=[];
    reclik.p.sigma2=[];
    if ~isempty(ri.p.sigma2)
      reclik.p.sigma2 = ri.p.sigma2;
    end
  else
    % Append to the record
    reclik.sigma2(ri,:)=lik.sigma2;
    if ~isempty(lik.p.sigma2)
      reclik.p.sigma2 = lik.p.sigma2.fh.recappend(reclik.p.sigma2, ri, lik.p.sigma2);
    end
  end
end
