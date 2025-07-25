function lik = lik_logit(varargin)
%LIK_LOGIT  Create a Logit likelihood structure 
%
%  Description
%    LIK = LIK_LOGIT creates Logit likelihood for classification
%    problem with class labels {-1,1}.
%       
%    The likelihood is defined as follows:
%               __ n
%      p(y|f) = || i=1 1/(1 + exp(-y_i*f_i) )
%    where f is the latent value vector.
%  
%    LIK = LIK_LOGIT('moment_method',METHOD) can be used to select
%      moment_method  - 'gh' 32 point Gauss-Hermite (default)
%                       or 'agk' adaptive Gauss-Kronord.
%  
%  See also
%    GP_SET, LIK_*
%
% Copyright (c) 2008-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_LOGIT';
  ip=iparser(ip,'addOptional','lik', [], @isstruct);
  ip=iparser(ip,'addParamValue','moment_method','gh', @(x) ismember(x,{'gh' 'agk'}));
  ip=iparser(ip,'parse',varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Logit';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Logit')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  % using Gibbs or not
  if init || ~ismember('moment_method',ip.UsingDefaults)
    lik.moment_method = ip.Results.moment_method;
  end

  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_logit_pak;
    lik.fh.unpak = @lik_logit_unpak;
    lik.fh.ll = @lik_logit_ll;
    lik.fh.llg = @lik_logit_llg;    
    lik.fh.llg2 = @lik_logit_llg2;
    lik.fh.llg3 = @lik_logit_llg3;
    switch lik.moment_method
      case 'agk'
        % adaptive Gauss-Kronord
        lik.fh.tiltedMoments = @lik_logit_tiltedMoments;
      otherwise
        % 32 point Gauss-Hermite
        lik.fh.tiltedMoments = @lik_logit_tiltedMoments_gh;
    end
    lik.fh.predy = @lik_logit_predy;
    lik.fh.invlink = @lik_logit_invlink;
    lik.fh.recappend = @lik_logit_recappend;
  end

end

function [w,s,h] = lik_logit_pak(lik)
%LIK_LOGIT_PAK  Combine likelihood parameters into one vector.
%
%  Description 
%    W = LIK_LOGIT_PAK(LIK) takes a likelihood structure LIK and
%    returns an empty verctor W. If Logit likelihood had
%    parameters this would combine them into a single row vector
%    W (see e.g. lik_negbin). This is a mandatory subfunction used 
%    for example in energy and gradient computations.
%       
%  See also
%    LIK_NEGBIN_UNPAK, GP_PAK

  w = []; s = {};h=[];
end


function [lik, w] = lik_logit_unpak(lik, w)
%LIK_LOGIT_UNPAK  Extract likelihood parameters from the vector.
%
%  Description
%    W = LIK_LOGIT_UNPAK(W, LIK) Doesn't do anything.
% 
%    If Logit likelihood had parameters this would extracts them
%    parameters from the vector W to the LIK structure. This is a 
%    mandatory subfunction used for example in energy and gradient 
%    computations.
%       
%  See also
%    LIK_LOGIT_PAK, GP_UNPAK

  lik=lik;
  w=w;
  
end

function ll = lik_logit_ll(lik, y, f, z)
%LIK_LOGIT_LL  Log likelihood
%
%  Description
%    E = LIK_LOGIT_LL(LIK, Y, F) takes a likelihood structure
%    LIK, class labels Y, and latent values F. Returns the log
%    likelihood, log p(y|f,z). This subfunction is also used in 
%    information criteria (DIC, WAIC) computations.
%
%  See also
%    LIK_LOGIT_LLG, LIK_LOGIT_LLG3, LIK_LOGIT_LLG2, GPLA_E

  if ~isempty(find(abs(y)~=1))
    error('lik_logit: The class labels have to be {-1,1}')
  end
  
  ll = sum(-log(1+exp(-y.*f)));
end


function llg = lik_logit_llg(lik, y, f, param, z)
%LIK_LOGIT_LLG  Gradient of the log likelihood
%
%  Description
%    G = LIK_LOGIT_LLG(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F. Returns
%    the gradient of the log likelihood with respect to PARAM. At the
%    moment PARAM can be 'param' or 'latent'.  This subfunction is 
%    needed when using Laplace approximation or MCMC for inference 
%    with non-Gaussian likelihoods.
%
%  See also
%    LIK_LOGIT_LL, LIK_LOGIT_LLG2, LIK_LOGIT_LLG3, GPLA_E
  
  if ~isempty(find(abs(y)~=1))
    error('lik_logit: The class labels have to be {-1,1}')
  end

  
  t  = (y+1)/2;
  PI = 1./(1+exp(-f));
  llg = t - PI;
  %llg = (y+1)/2 - 1./(1+exp(-f));      
end


function llg2 = lik_logit_llg2(lik, y, f, param, z)
%LIK_LOGIT_LLG2  Second gradients of the log likelihood
%
%  Description        
%    LLG2 = LIK_LOGIT_LLG2(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F. Returns
%    the Hessian of the log likelihood with respect to PARAM. At
%    the moment PARAM can be only 'latent'. LLG2 is a vector with
%    diagonal elements of the Hessian matrix (off diagonals are
%    zero). This subfunction is needed when using Laplace approximation 
%    or EP for inference with non-Gaussian likelihoods.
%
%   See also
%   LIK_LOGIT_LL, LIK_LOGIT_LLG, LIK_LOGIT_LLG3, GPLA_E

  PI = 1./(1+exp(-f));
  llg2 = -PI.*(1-PI);        
end    

function llg3 = lik_logit_llg3(lik, y, f, param, z)
%LIK_LOGIT_LLG3  Third gradients of the log likelihood
%
%  Description
%    LLG3 = LIK_LOGIT_LLG3(LIK, Y, F, PARAM) takes a likelihood
%    structure LIK, class labels Y, and latent values F and
%    returns the third gradients of the log likelihood with
%    respect to PARAM. At the moment PARAM can be only 'latent'. 
%    LLG3 is a vector with third gradients. This subfunction is 
%    needed when using Laplace approximation for inference with 
%    non-Gaussian likelihoods.
%
%  See also
%    LIK_LOGIT_LL, LIK_LOGIT_LLG, LIK_LOGIT_LLG2, GPLA_E, GPLA_G
  
  if ~isempty(find(abs(y)~=1))
    error('lik_logit: The class labels have to be {-1,1}')
  end

  t  = (y+1)/2;
  PI = 1./(1+exp(-f));
  llg3 = -PI.*(1-PI).*(1-2*PI);        
end

function [logM_0, m_1, sigm2hati1, m_3, m_4] = lik_logit_tiltedMoments_gh(lik, y, i1, sigma2_i, myy_i, z)
%LIK_LOGIT_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_LOGIT_TILTEDMOMENTS(LIK, Y, I, S2, MYY)
%    takes a likelihood structure LIK, class labels Y, index I
%    and cavity variance S2 and mean MYY. Returns the zeroth
%    moment M_0, mean M_1 and variance M_2 of the posterior
%    marginal (see Rasmussen and Williams (2006): Gaussian
%    processes for Machine Learning, page 55). This subfunction 
%    is needed when using EP for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    GPEP_E
  
% don't check this here, because this function is called so often by EP
%  if ~isempty(find(abs(y)~=1))
%    error('lik_logit: The class labels have to be {-1,1}')
%  end

% Use 32 point Gauss-Hermite quadrature
% http://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature
% fewer points might be sufficient, but this is more safe
  xhermite=[-7.12581390983 -6.40949814928 -5.81222594946 -5.27555098664 -4.77716450334 -4.30554795347 -3.85375548542 -3.41716749282 -2.99249082501 -2.57724953773 -2.16949918361 -1.76765410946 -1.37037641095 -0.97650046359 -0.584978765436 -0.194840741569 0.194840741569 0.584978765436 0.97650046359 1.37037641095 1.76765410946 2.16949918361 2.57724953773 2.99249082501 3.41716749282 3.85375548542 4.30554795347 4.77716450334 5.27555098664 5.81222594946 6.40949814928 7.12581390983];
  whermite=[7.31067642754e-023 9.23173653482e-019 1.19734401957e-015 4.21501019491e-013 5.93329148347e-011 4.09883215841e-009 1.5741677944e-007 3.65058512533e-006 5.41658405999e-005 0.000536268365495 0.00365489032677 0.0175534288315 0.0604581309559 0.151269734077 0.277458142303 0.375238352593 0.375238352593 0.277458142303 0.151269734077 0.0604581309559 0.0175534288315 0.00365489032677 0.000536268365495 5.41658405999e-005 3.65058512533e-006 1.5741677944e-007 4.09883215841e-009 5.93329148347e-011 4.21501019491e-013 1.19734401957e-015 9.23173653482e-019 7.31067642754e-023];
  yy=y(i1);
  xtransformed=bsxfun(@plus,bsxfun(@times,sqrt(2*sigma2_i),xhermite),myy_i);
  h=exp(-log(1+exp(bsxfun(@times,-yy,xtransformed))));
  wh=bsxfun(@times,whermite,h);
  m_0=nansum(wh,2)/sqrt(pi);
  if any(isnan(m_0))
    logM_0=NaN;
    m_1=NaN;
    m_2=NaN;
    return
  end
  logM_0 = log(m_0);
  if nargout>1
    whx=bsxfun(@times,wh,xtransformed);
    m_1=sum(whx,2)/sqrt(pi)./m_0;
    if nargout>2
      whx2=bsxfun(@times,whx,xtransformed);
      m_2=sum(whx2,2)/sqrt(pi)./m_0;
      sigm2hati1 = m_2 - m_1.^2;
      trouble=sigm2hati1 >= sigma2_i;
      sigm2hati1(trouble)=sigma2_i(trouble)-eps*10;
    end
  end
end

function [logM_0, m_1, sigm2hati1, m_3, m_4] = lik_logit_tiltedMoments(lik, y, i1, sigma2_i, myy_i, z)
%LIK_LOGIT_TILTEDMOMENTS    Returns the marginal moments for EP algorithm
%
%  Description
%    [M_0, M_1, M2] = LIK_LOGIT_TILTEDMOMENTS(LIK, Y, I, S2, MYY)
%    takes a likelihood structure LIK, class labels Y, index I
%    and cavity variance S2 and mean MYY. Returns the zeroth
%    moment M_0, mean M_1 and variance M_2 of the posterior
%    marginal (see Rasmussen and Williams (2006): Gaussian
%    processes for Machine Learning, page 55). This subfunction 
%    is needed when using EP for inference with non-Gaussian 
%    likelihoods.
%
%  See also
%    GPEP_E
  
% don't check this here, because this function is called so often by EP
%  if ~isempty(find(abs(y)~=1))
%    error('lik_logit: The class labels have to be {-1,1}')
%  end
  
% Use adaptive Gauss-Kronord quadrature
  yy = y(i1);
  logM_0=zeros(size(yy));
  m_1=zeros(size(yy));
  sigm2hati1=zeros(size(yy));
  
  for i=1:length(i1)
    if isscalar(sigma2_i)
      sigma2ii = sigma2_i;
    else
      sigma2ii = sigma2_i(i);
    end
    
    % get a function handle of an unnormalized tilted distribution
    % (likelihood * cavity = Logit * Gaussian)
    % and useful integration limits
    [tf,minf,maxf]=init_logit_norm(yy(i),myy_i(i),sigma2ii);
    
    if isnan(minf) || isnan(maxf)
      logM_0(i)=NaN; m_1(i)=NaN; sigm2hati1(i)=NaN;
      continue
    end
    
    % Integrate with an adaptive Gauss-Kronrod quadrature
    % (Rasmussen and Nickish use in GPML interpolation between
    % a cumulative Gaussian scale mixture and linear tail
    % approximation, which could be faster, but quadrature also
    % takes only a fraction of the time EP uses overall, so no
    % need to change...)
    RTOL = 1.e-6;
    ATOL = 1.e-10;
    if nargout < 4
      [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
    else
      [m_0, m_1(i), m_2, m_3, m_4] = quad_moments(tf, minf, maxf, RTOL, ATOL);
    end
    if isnan(m_0)
      logM_0=NaN;
      return
    end
    sigm2hati1(i) = m_2 - m_1(i).^2;
    
    % If the second central moment is less than cavity variance
    % integrate more precisely. Theoretically should be
    % sigm2hati1 < sigm2_i.
    if sigm2hati1(i) >= sigma2ii
      ATOL = ATOL.^2;
      RTOL = RTOL.^2;
      [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
      sigm2hati1(i) = m_2 - m_1(i).^2;
      if sigm2hati1(i) >= sigma2ii
        %warning('lik_logit_tilted_moments: sigm2hati1 >= sigm2_i');
        sigm2hati1(i)=sigma2ii-eps;
      end
    end
    logM_0(i) = log(m_0);
  end
end

function [lpy, Ey, Vary] = lik_logit_predy(lik, Ef, Varf, yt, zt)
%LIK_LOGIT_PREDY  Returns the predictive mean, variance and density of y
%
%  Description         
%    LPY = LIK_LOGIT_PREDY(LIK, EF, VARF, YT)
%    Returns logarithm of the predictive density of YT, that is 
%        p(yt | y) = \int p(yt | f) p(f|y) df.
%    This requires also the class labels YT. This subfunction 
%    is needed when computing posterior predictive distributions 
%    for future observations.
%
%    [LPY, EY, VARY] = LIK_LOGIT_PREDY(LIK, EF, VARF) takes a
%    likelihood structure LIK, posterior mean EF and posterior
%    Variance VARF of the latent variable and returns also the
%    posterior predictive mean EY and variance VARY of the
%    observations related to the latent variables. This subfunction 
%    is needed when computing posterior predictive distributions for 
%    future observations.
%
%  See also 
%    GPLA_PRED, GPEP_PRED, GPMC_PRED
  
    if nargout > 1
      py1 = zeros(length(Ef),1);
      for i1=1:length(Ef)
        myy_i = Ef(i1);
        sigm_i = sqrt(Varf(i1));
        minf=myy_i-6*sigm_i;
        maxf=myy_i+6*sigm_i;
        F  = @(f)1./(1+exp(-f)).*norm_pdf(f,myy_i,sigm_i);
        py1(i1) = quadgk(F,minf,maxf);
      end
      Ey = 2*py1-1;
      Vary = 1-(2*py1-1).^2;
    end
  

    if ~isempty(find(abs(yt)~=1))
      error('lik_logit: The class labels have to be {-1,1}')
    end
    % Quadrature integration                                    
    lpy = zeros(length(yt),1);
    if (size(Ef,2) > 1) && (size(Ef,2) > 1) && size(yt,2) == 1
      % Approximate integral with sum of grid points when using corrected
      % marginal posterior pf
      for i1=1:length(yt)
        py = arrayfun(@(f) exp(lik.fh.ll(lik, yt(i1), f, [])), Ef(i1,:));
        pf = Varf(i1,:)./sum(Varf(i1,:));
        lpy(i1) = log(sum(py.*pf));
      end
    else
      for i1 = 1:length(yt)
        % get a function handle of the likelihood times posterior
        % (likelihood * posterior = Poisson * Gaussian)
        % and useful integration limits
        [pdf,minf,maxf]=init_logit_norm(...
          yt(i1),Ef(i1),Varf(i1));
        % integrate over the f to get posterior predictive distribution
        lpy(i1) = log(quadgk(pdf, minf, maxf));
      end
    end
end

function [df,minf,maxf] = init_logit_norm(yy,myy_i,sigm2_i)
%INIT_LOGIT_NORM
%
%  Description
%    Return function handle to a function evaluating Logit *
%    Gaussian which is used for evaluating (likelihood * cavity)
%    or (likelihood * posterior) Return also useful limits for
%    integration. This is private function for lik_logit. This 
%    subfunction is needed by subfunctions tiltedMoments, siteDeriv
%    and predy.
%  
% See also
%   LIK_LOGIT_TILTEDMOMENTS, LIK_LOGIT_PREDY
  
% avoid repetitive evaluation of constant part
  ldconst = -log(sigm2_i)/2 -log(2*pi)/2;
  
  % Create function handle for the function to be integrated
  df = @(f) logit_norm(f, ldconst, yy, myy_i, sigm2_i);
  % use log to avoid underflow, and derivates for faster search
  ld = @log_logit_norm;
  ldg = @log_logit_norm_g;
  ldg2 = @log_logit_norm_g2;

  % Set the limits for integration
  % Logit likelihood is log-concave so the logit_norm
  % function is unimodal, which makes things easier
  
  % approximate guess for the location of the mode
  if sign(myy_i)==sign(yy)
    % the log likelihood is flat on this side
    modef = myy_i;
  else
    % the log likelihood is approximately yy*f on this side
    modef=sign(myy_i)*max(abs(myy_i)-sigm2_i,0);
  end
  % find the mode of the integrand using Newton iterations
  % few iterations is enough, since the first guess in the right direction
  niter=2;       % number of Newton iterations
  mindelta=1e-6; % tolerance in stopping Newton iterations
  for ni=1:niter
    g=ldg(modef, ldconst, yy, myy_i, sigm2_i);
    h=ldg2(modef, ldconst, yy, myy_i, sigm2_i);
    delta=-g/h;
    modef=modef+delta;
    if abs(delta)<mindelta
      break
    end
  end
  % integrand limits based on Gaussian approximation at mode
  modes=sqrt(-1/h);
  minf=modef-8*modes;
  maxf=modef+8*modes;
  modeld=ld(modef, ldconst, yy, myy_i, sigm2_i);
  if isinf(modeld) || isnan(modeld)
    minf=NaN;maxf=NaN;
    return
  end
  iter=0;
  % check that density at end points is low enough
  lddiff=20; % min difference in log-density between mode and end-points
  minld=ld(minf, ldconst, yy, myy_i, sigm2_i);
  step=1;
  while minld>(modeld-lddiff)
    minf=minf-step*modes;
    minld=ld(minf, ldconst, yy, myy_i, sigm2_i);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_logit -> init_logit_norm: ' ...
             'integration interval minimun not found ' ...
             'even after looking hard!'])
    end
  end
  maxld=ld(maxf, ldconst, yy, myy_i, sigm2_i);
  iter=0;
  step=1;
  while maxld>(modeld-lddiff)
    maxf=maxf+step*modes;
    maxld=ld(maxf, ldconst, yy, myy_i, sigm2_i);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_logit -> init_logit_norm: ' ...
             'integration interval maximum not found ' ...
             'even after looking hard!'])
    end
  end
end
  function integrand = logit_norm(f, ldconst, yy, myy_i, sigm2_i)
  % Logit * Gaussian
    integrand = exp(ldconst ...
                    -log(1+exp(-yy.*f)) ...
                    -0.5*(f-myy_i).^2./sigm2_i);
  end
  
  function log_int = log_logit_norm(f, ldconst, yy, myy_i, sigm2_i)
  % log(Logit * Gaussian)
  % log_logit_norm is used to avoid underflow when searching
  % integration interval
    log_int = ldconst ...
              -log(1+exp(-yy.*f)) ...
              -0.5*(f-myy_i).^2./sigm2_i;
  end
  
  function g = log_logit_norm_g(f, ldconst, yy, myy_i, sigm2_i)
  % d/df log(Logit * Gaussian)
  % derivative of log_logit_norm
    g = yy./(exp(f*yy)+1)...
        + (myy_i - f)./sigm2_i;
  end
  
  function g2 = log_logit_norm_g2(f, ldconst, yy, myy_i, sigm2_i)
  % d^2/df^2 log(Logit * Gaussian)
  % second derivate of log_logit_norm
    a=exp(f*yy);
    g2 = -a*(yy./(a+1)).^2 ...
         -1/sigm2_i;
  end
 
function p = lik_logit_invlink(lik, f, z)
%LIK_LOGIT_INVLINK  Returns values of inverse link function
%             
%  Description 
%    P = LIK_LOGIT_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values of inverse link function P.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_LOGIT_LL, LIK_LOGIT_PREDY
  
  p = logitinv(f);
end

function reclik = lik_logit_recappend(reclik, ri, lik)
%RECAPPEND  Append the parameters to the record
%
%  Description 
%    RECLIK = GPCF_LOGIT_RECAPPEND(RECLIK, RI, LIK) takes a
%    likelihood record structure RECLIK, record index RI and
%    likelihood structure LIK with the current MCMC samples of
%    the parameters. Returns RECLIK which contains all the old
%    samples and the current samples from LIK. This subfunction
%    is needed when using MCMC sampling (gp_mc).
% 
%  See also
%    GP_MC

  if nargin == 2
    reclik.type = 'Logit';

    % Set the function handles
    reclik.fh.pak = @lik_logit_pak;
    reclik.fh.unpak = @lik_logit_unpak;
    reclik.fh.ll = @lik_logit_ll;
    reclik.fh.llg = @lik_logit_llg;    
    reclik.fh.llg2 = @lik_logit_llg2;
    reclik.fh.llg3 = @lik_logit_llg3;
    switch ri.moment_method
      case 'agk'
        % adaptive Gauss-Kronord
        reclik.fh.tiltedMoments = @lik_logit_tiltedMoments;
      otherwise
        % 32 point Gauss-Hermite
        reclik.fh.tiltedMoments = @lik_logit_tiltedMoments_gh;
    end
    reclik.fh.predy = @lik_logit_predy;
    reclik.fh.invlink = @lik_logit_invlink;
    reclik.fh.recappend = @lik_logit_recappend;
  end
end

