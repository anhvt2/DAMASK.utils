function lik = lik_zinegbin(varargin)
%LIK_ZINEGBIN    Create a zero-inflated Negative-binomial likelihood structure 
%
%  Description
%    LIK = LIK_ZINEGBIN('PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    creates a zero-inflated Negative-binomial likelihood structure in
%    which the named parameters have the specified values. Any unspecified
%    parameters are set to default values.  
%  
%    LIK = LIK_ZINEGBIN(LIK,'PARAM1',VALUE1,'PARAM2,VALUE2,...)
%    modify a likelihood structure with the named parameters
%    altered with the specified values.
%
%    Parameters for a zero-inflated Negative-binomial likelihood [default]
%      disper       - dispersion parameter r [10]
%      disper_prior - prior for disper [prior_logunif]
%  
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%    The likelihood is defined as follows:
%     
%      p + (1-p)*NegBin(y|y=0),    when y=0
%          (1-p)*NegBin(y|y>0),    when y>0,
%
%      where the probability p is given by a binary classifier with Logit
%      likelihood and NegBin is the Negative-binomial distribution
%      parametrized for the i'th observation as
%                  
%      NegBin(y_i) =   [ (r/(r+mu_i))^r * gamma(r+y_i)
%                        / ( gamma(r)*gamma(y_i+1) )
%                        * (mu/(r+mu_i))^y_i ]
%
%    where mu_i = z_i*exp(f_i) and r is the dispersion parameter.
%    z is a vector of expected mean and f the latent value vector
%    whose components are transformed to relative risk
%    exp(f_i). 
%
%    The latent value vector f=[f1^T f2^T]^T has length 2*N, where N is the
%    number of observations. The latents f1 are associated with the
%    classification process and the latents f2 with Negative-binomial count
%    process.
%
%    When using the Zinegbin likelihood you can give the
%    vector z as an extra parameter to each function that requires
%    also y. For example, you can call gp_optim as follows:
%      gp_optim(gp, x, y, 'z', z)
%    If z is not given or it is empty, then z_i=1 is used.
%
%  See also
%    GP_SET, LIK_*, PRIOR_*
%
% Copyright (c) 2007-2010 Jarno Vanhatalo & Jouni Hartikainen
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2011 Jaakko Riihimäki

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'LIK_ZINEGBIN';
  ip=iparser(ip,'addOptional','lik', [], @isstruct);
  ip=iparser(ip,'addParamValue','disper',10, @(x) isscalar(x) && x>0);
  ip=iparser(ip,'addParamValue','disper_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
  ip=iparser(ip,'parse',varargin{:});
  lik=ip.Results.lik;
  
  if isempty(lik)
    init=true;
    lik.type = 'Zinegbin';
    lik.nondiagW=true;
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Zinegbin')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end

  % Initialize parameters
  if init || ~ismember('disper',ip.UsingDefaults)
    lik.disper = ip.Results.disper;
  end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('disper_prior',ip.UsingDefaults)
    lik.p.disper=ip.Results.disper_prior;
  end
  
  if init
    % Set the function handles to the nested functions
    lik.fh.pak = @lik_zinegbin_pak;
    lik.fh.unpak = @lik_zinegbin_unpak;
    lik.fh.lp = @lik_zinegbin_lp;
    lik.fh.lpg = @lik_zinegbin_lpg;
    lik.fh.ll = @lik_zinegbin_ll;
    lik.fh.llg = @lik_zinegbin_llg;    
    lik.fh.llg2 = @lik_zinegbin_llg2;
    lik.fh.llg3 = @lik_zinegbin_llg3;
    lik.fh.predy = @lik_zinegbin_predy;
    lik.fh.invlink = @lik_zinegbin_invlink;
    lik.fh.recappend = @lik_zinegbin_recappend;
  end
end

  function [w,s,h] = lik_zinegbin_pak(lik)
  %LIK_ZINEGBIN_PAK  Combine likelihood parameters into one vector.
  %
  %  Description 
  %    W = LIK_ZINEGBIN_PAK(LIK) takes a likelihood structure LIK and
  %    combines the parameters into a single row vector W. This is a 
  %    mandatory subfunction used for example in energy and gradient 
  %    computations.
  %     
  %       w = log(lik.disper)
  %
  %   See also
  %   LIK_ZINEGBIN_UNPAK, GP_PAK
    
    w=[];s={};h=[];
    if ~isempty(lik.p.disper)
      w = log(lik.disper);
      s = [s; 'log(zinegbin.disper)'];
      h = 0;
      [wh, sh, hh] = feval(lik.p.disper.fh.pak, lik.p.disper);
      w = [w wh];
      s = [s; sh];
      h = [h hh];
    end
  end


  function [lik, w] = lik_zinegbin_unpak(lik, w)
  %LIK_ZINEGBIN_UNPAK  Extract likelihood parameters from the vector.
  %
  %  Description
  %    [LIK, W] = LIK_ZINEGBIN_UNPAK(W, LIK) takes a likelihood
  %    structure LIK and extracts the parameters from the vector W
  %    to the LIK structure. This is a mandatory subfunction used for 
  %    example in energy and gradient computations.
  %     
  %   Assignment is inverse of  
  %       w = log(lik.disper)
  %
  %   See also
  %   LIK_ZINEGBIN_PAK, GP_UNPAK

    if ~isempty(lik.p.disper)
      lik.disper = exp(w(1));
      w = w(2:end);
      [p, w] = feval(lik.p.disper.fh.unpak, lik.p.disper, w);
      lik.p.disper = p;
    end
  end


  function lp = lik_zinegbin_lp(lik, varargin)
  %LIK_ZINEGBIN_LP  log(prior) of the likelihood parameters
  %
  %  Description
  %    LP = LIK_ZINEGBIN_LP(LIK) takes a likelihood structure LIK and
  %    returns log(p(th)), where th collects the parameters. This 
  %    subfunction is needed when there are likelihood parameters.
  %
  %  See also
  %    LIK_ZINEGBIN_LLG, LIK_ZINEGBIN_LLG3, LIK_ZINEGBIN_LLG2, GPLA_E
    

  % If prior for dispersion parameter, add its contribution
    lp=0;
    if ~isempty(lik.p.disper)
      lp = feval(lik.p.disper.fh.lp, lik.disper, lik.p.disper) +log(lik.disper);
    end
    
  end

  
  function lpg = lik_zinegbin_lpg(lik)
  %LIK_ZINEGBIN_LPG  d log(prior)/dth of the likelihood 
  %                parameters th
  %
  %  Description
  %    E = LIK_ZINEGBIN_LPG(LIK) takes a likelihood structure LIK and
  %    returns d log(p(th))/dth, where th collects the parameters. This
  %    subfunction is needed when there are likelihood parameters.
  %
  %  See also
  %    LIK_ZINEGBIN_LLG, LIK_ZINEGBIN_LLG3, LIK_ZINEGBIN_LLG2, GPLA_G
    
    lpg=[];
    if ~isempty(lik.p.disper)            
      % Evaluate the gprior with respect to disper
      ggs = feval(lik.p.disper.fh.lpg, lik.disper, lik.p.disper);
      lpg = ggs(1).*lik.disper + 1;
      if length(ggs) > 1
        lpg = [lpg ggs(2:end)];
      end
    end
  end  
  
  function ll = lik_zinegbin_ll(lik, y, ff, z)
  %LIK_ZINEGBIN_LL  Log likelihood
  %
  %  Description
  %    LL = LIK_ZINEGBIN_LL(LIK, Y, F, Z) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z, and
  %    latent values F. Returns the log likelihood, log p(y|f,z).
  %    This subfunction is needed when using Laplace approximation
  %    or MCMC for inference with non-Gaussian likelihoods. This
  %    subfunction is also used in information criteria (DIC, WAIC)
  %    computations.
  %
  %  See also
  %    LIK_ZINEGBIN_LLG, LIK_ZINEGBIN_LLG3, LIK_ZINEGBIN_LLG2, GPLA_E
    
    if numel(z)==0
      z=1;
    end
    
    f=ff(:);
    n=size(y,1);
    f1=f(1:n);
    f2=f((n+1):2*n);
    y0ind=y==0;
    yind=y>0;
    
    r = lik.disper;
    m = exp(f2).*z;
    expf1=exp(f1);
    
    % for y = 0
    lly0 = sum(-log(1+expf1(y0ind)) + log( expf1(y0ind) + (r./(r+m(y0ind))).^r ));
    % for y > 0
    lly = sum(-log(1+expf1(yind)) + r.*(log(r) - log(r+m(yind))) + gammaln(r+y(yind)) - gammaln(r) - gammaln(y(yind)+1) + y(yind).*(log(m(yind)) - log(r+m(yind))));
    
    ll=lly0+lly;
  end

  function llg = lik_zinegbin_llg(lik, y, ff, param, z)
  %LIK_ZINEGBIN_LLG  Gradient of the log likelihood
  %
  %  Description 
  %    LLG = LIK_ZINEGBIN_LLG(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z and
  %    latent values F. Returns the gradient of the log likelihood
  %    with respect to PARAM. At the moment PARAM can be 'param' or
  %    'latent'. This subfunction is needed when using Laplace 
  %    approximation or MCMC for inference with non-Gaussian likelihoods.
  %
  %  See also
  %    LIK_ZINEGBIN_LL, LIK_ZINEGBIN_LLG2, LIK_ZINEGBIN_LLG3, GPLA_E

    if numel(z)==0
      z=1;
    end

    f=ff(:);
	  n=size(y,1);
    f1=f(1:n);
    f2=f((n+1):2*n);
    y0ind=y==0;
    yind=y>0;
    
    r = lik.disper;
    m = exp(f2).*z;
    expf1=exp(f1);
    
    switch param
      case 'param'
          
        % syms e r m
        % simplify(diff(log( e + (r./(r+m)).^r ),r))
        
        m0=m(y0ind);
        llg1=sum( ((r./(m0 + r)).^r.*(m0 + m0.*log(r./(m0 + r)) + r.*log(r./(m0 + r))))./((expf1(y0ind) + (r./(m0 + r)).^r).*(m0 + r)) );
        
        % Derivative using the psi function
        llg2 = sum(1 + log(r./(r+m(yind))) - (r+y(yind))./(r+m(yind)) + psi(r + y(yind)) - psi(r));
        
        llg = llg1 + llg2;
        % correction for the log transformation
        llg = llg.*lik.disper;
      case 'latent'
        
        llg1=zeros(n,1);
        llg2=zeros(n,1);
        
        llg1(y0ind)=-expf1(y0ind)./(1+expf1(y0ind)) + expf1(y0ind)./(expf1(y0ind)+ (r./(r+m(y0ind))).^r);
        llg2(y0ind)=-1./( expf1(y0ind) + (r./(r+m(y0ind))).^r ) .* (r./(r+m(y0ind))).^(r-1) .* (m(y0ind).*r.^2./((r+m(y0ind)).^2));
        
        llg1(yind)=-expf1(yind)./(1+expf1(yind));
        llg2(yind)=y(yind) - (r+y(yind)).*m(yind)./(r+m(yind));
        
        llg=[llg1; llg2];
    end
  end

  function llg2 = lik_zinegbin_llg2(lik, y, ff, param, z)
  %LIK_ZINEGBIN_LLG2  Second gradients of the log likelihood
  %
  %  Description        
  %    LLG2 = LIK_ZINEGBIN_LLG2(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z, and
  %    latent values F. Returns the Hessian of the log likelihood
  %    with respect to PARAM. At the moment PARAM can be only
  %    'latent'. Second gradients form a matrix of size 2N x 2N as
  %    [diag(LLG2_11) diag(LLG2_12); diag(LLG2_12) diag(LLG2_22)],
  %    but the function returns only vectors of diagonal elements as
  %    LLG2 = [LLG2_11 LLG2_12; LLG2_12 LLG2_22] (2Nx2 matrix) since off
  %    diagonals of the blocks are zero. This subfunction is needed when 
  %    using Laplace approximation or EP for inference with non-Gaussian 
  %    likelihoods.
  %
  %  See also
  %    LIK_ZINEGBIN_LL, LIK_ZINEGBIN_LLG, LIK_ZINEGBIN_LLG3, GPLA_E

    if numel(z)==0
      z=1;
    end

    f=ff(:);
    
    n=size(y,1);
    f1=f(1:n);
    f2=f((n+1):2*n);
    y0ind=y==0;
    yind=y>0;
    
    r = lik.disper;
    m = exp(f2).*z;
    expf1=exp(f1);

    switch param
      case 'param'
        
      case 'latent'
          
        llg2_11=zeros(n,1);
        llg2_12=zeros(n,1);
        llg2_22=zeros(n,1);
        
        rmuy0=(r./(r+m(y0ind))).^r;
        llg2_11(y0ind)=-expf1(y0ind)./(1+expf1(y0ind)).^2 + expf1(y0ind).*rmuy0./(expf1(y0ind)+rmuy0).^2;
        llg2_12(y0ind)=(r./(r+m(y0ind))).^(r-1).*(m(y0ind).*r.^2./(r+m(y0ind)).^2).*expf1(y0ind)./(expf1(y0ind)+rmuy0).^2;
        llg2_22(y0ind)=-1./(expf1(y0ind)+rmuy0).*( (r./(r+m(y0ind))).^(2*r-2).*(m(y0ind).*r.^2./(r+m(y0ind)).^2).^2./(expf1(y0ind)+rmuy0) + (r-1).*(r./(r+m(y0ind))).^(r-2).*(-m(y0ind).^2.*r.^3./(r+m(y0ind)).^4) + (r./(r+m(y0ind))).^(r-1).*(m(y0ind).*r.^2.*(r+m(y0ind))-2*m(y0ind).^2.*r.^2)./(r+m(y0ind)).^3);
        
        llg2_11(yind)=-expf1(yind)./(1+expf1(yind)).^2;
        llg2_22(yind)=-m(yind).*(r.^2 + y(yind).*r)./(r+m(yind)).^2;
        
%         llg2 = [diag(llg2_11) diag(llg2_12); diag(llg2_12) diag(llg2_22)];
        llg2 = [llg2_11 llg2_12; llg2_12 llg2_22];
        
%         R=[];
%         nl=2;
%         pi_mat=zeros(nl*n, n);
%         llg2_12sq=sqrt(llg2_12);
%         for i1=1:nl
%             pi_mat((1+(i1-1)*n):(nl*n+1):end)=llg2_12sq;
%         end
%         pi_vec=repmat(llg2_12,nl,1)-[llg2_11; llg2_22];
        %D=diag(pi_vec);
        %llg2=-D+pi_mat*pi_mat';
        %imagesc((-diag(pi_vec)+pi_mat*pi_mat') - llg2),colorbar
        
      case 'latent+param'
        
        %syms e r m
        %simplify(diff(e./(e + (r./(r+m)).^r),r))
        %simplify(diff(-1/( e + (r/(r+m))^r ) * (r/(r+m))^(r-1) * (m*r^2/((r+m)^2)),r))
        
        llg2_1=zeros(n,1);
        llg2_2=zeros(n,1);
        
        m0=m(y0ind);
        e0=expf1(y0ind);
        llg2_1(y0ind)=-(e0.*(r./(m0 + r)).^r.*(m0 + m0.*log(r./(m0 + r)) + r.*log(r./(m0 + r))))./((e0 + (r./(m0 + r)).^r).^2.*(m0 + r));
        llg2_2(y0ind)=-(m0.*(r./(m0 + r)).^r.*(m0.*(r./(m0 + r)).^r + e0.*m0 + e0.*r.^2.*log(r./(m0 + r)) + e0.*m0.*r + e0.*m0.*r.*log(r./(m0 + r))))./((e0 + (r./(m0 + r)).^r).^2.*(m0 + r).^2);
   
        llg2_2(yind)= (y(yind).*m(yind) - m(yind).^2)./(r+m(yind)).^2;
        
        llg2=[llg2_1; llg2_2];
        
        % correction due to the log transformation
        llg2 = llg2.*lik.disper;
    end
  end    
  
  function llg3 = lik_zinegbin_llg3(lik, y, ff, param, z)
  %LIK_ZINEGBIN_LLG3  Third gradients of the log likelihood
  %
  %  Description
  %    LLG3 = LIK_ZINEGBIN_LLG3(LIK, Y, F, PARAM) takes a likelihood
  %    structure LIK, incedence counts Y, expected counts Z and
  %    latent values F and returns the third gradients of the log
  %    likelihood with respect to PARAM. At the moment PARAM can be
  %    only 'latent'. LLG3 is a 2-by-2-by-2-by-N array of with third
  %    gradients, where LLG3(:,:,1,i) is the third derivative wrt f1 for
  %    the i'th observation and LLG3(:,:,2,i) is the third derivative wrt
  %    f2 for the i'th observation. This subfunction is needed when using 
  %    Laplace approximation for inference with non-Gaussian likelihoods.
  %
  %  See also
  %    LIK_ZINEGBIN_LL, LIK_ZINEGBIN_LLG, LIK_ZINEGBIN_LLG2, GPLA_E, GPLA_G

    if numel(z)==0
      z=1;
    end

    f=ff(:);
    
    n=size(y,1);
    f1=f(1:n);
    f2=f((n+1):2*n);
    y0ind=y==0;
    yind=y>0;
    
    r = lik.disper;
    m = exp(f2).*z;
    expf1=exp(f1);
    
    switch param
      case 'param'
        
      case 'latent'
      nl=2;
      llg3=zeros(nl,nl,nl,n);
      
      % 11
      %simplify(diff(-exp(f1)/(1+exp(f1))^2 + exp(f1)*(r/(r+exp(f2)*z))^r/(exp(f1)+(r/(r+exp(f2)*z))^r)^2,f1))          
      %simplify(diff(-exp(f1)/(1+exp(f1))^2 + exp(f1)*(r/(r+exp(f2)*z))^r/(exp(f1)+(r/(r+exp(f2)*z))^r)^2,f2))
      
      % 12/21
      % simplify(diff((r/(r+exp(f2)*z))^(r-1)*(exp(f2)*z*r^2/(r+exp(f2)*z)^2)*exp(f1)/(exp(f1)+(r/(r+exp(f2)*z))^r)^2,f1))
      % simplify(diff((r/(r+exp(f2)*z))^(r-1)*(exp(f2)*z*r^2/(r+exp(f2)*z)^2)*exp(f1)/(exp(f1)+(r/(r+exp(f2)*z))^r)^2,f2))
      
      % 22
      % simplify(diff(-1/(exp(f1)+(r/(r+z*exp(f2)))^r)*((r/(r+z*exp(f2)))^(2*r-2)*(z*exp(f2)*r^2/(r+z*exp(f2))^2)^2/(exp(f1)+(r/(r+z*exp(f2)))^r) + (r-1)*(r/(r+z*exp(f2)))^(r-2)*(-z*exp(f2)^2*r^3/(r+z*exp(f2))^4) + (r/(r+z*exp(f2)))^(r-1)*(z*exp(f2)*r^2*(r+z*exp(f2))-2*z*exp(f2)^2*r^2)/(r+z*exp(f2))^3),f1))
      % simplify(diff(-1/(exp(f1)+(r/(r+z*exp(f2)))^r)*((r/(r+z*exp(f2)))^(2*r-2)*(z*exp(f2)*r^2/(r+z*exp(f2))^2)^2/(exp(f1)+(r/(r+z*exp(f2)))^r) + (r-1)*(r/(r+z*exp(f2)))^(r-2)*(-z*exp(f2)^2*r^3/(r+z*exp(f2))^4) + (r/(r+z*exp(f2)))^(r-1)*(z*exp(f2)*r^2*(r+z*exp(f2))-2*z*exp(f2)^2*r^2)/(r+z*exp(f2))^3),f2))
      % with symbolic math toolbox:
      % simplify(diff(simplify(diff(simplify(diff((-log(1+exp(f1)) + log( exp(f1) + (r./(r+z*exp(f2))).^r )),f2)),f2)),f1))
      
      expf2=exp(f2);
      m0=m(y0ind);
      z0=z(y0ind);
      expf1y0ind=expf1(y0ind);
      expf1y0ind2=expf1y0ind.^2;
      expf2y0ind=expf2(y0ind);
      
      % y=0:
      % thrid derivative derivative wrt f1 (11)
      llg3(1,1,1,y0ind) = (2.*expf1y0ind2)./(expf1y0ind + 1).^3 - expf1y0ind./(expf1y0ind + 1).^2 - (2.*expf1y0ind2.*(r./(r + m0)).^r)./(expf1y0ind + (r./(r + m0)).^r).^3 + (expf1y0ind.*(r./(r + m0)).^r)./(expf1y0ind + (r./(r + m0)).^r).^2;
      % thrid derivative derivative wrt f2 (11)
      llg3(1,1,2,y0ind)=-(r.*m0.*expf1y0ind.*(expf1y0ind - (r./(r + m0)).^r).*(r./(r + m0)).^r)./((r + m0).*(expf1y0ind + (r./(r + m0)).^r).^3);
        
      % thrid derivative derivative wrt f1 (12/21)
      llg3(1,2,1,y0ind) = -(r.*m0.*expf1y0ind.*(expf1y0ind - (r./(r + m0)).^r).*(r./(r + m0)).^r)./((r + m0).*(expf1y0ind + (r./(r + m0)).^r).^3);
      llg3(2,1,1,y0ind) = llg3(1,2,1,y0ind);
      % thrid derivative derivative wrt f2 (12/21)
      llg3(1,2,2,y0ind) = (r.^2.*m0.*expf1y0ind.*(r./(r + m0)).^r.*(expf1y0ind + (r./(r + m0)).^r + m0.*(r./(r + m0)).^r - expf1y0ind.*m0))./((r + m0).^2.*(expf1y0ind + (r./(r + m0)).^r).^3);
      llg3(2,1,2,y0ind) = llg3(1,2,2,y0ind);
      
      % thrid derivative derivative wrt f1 (22)
      llg3(2,2,1,y0ind) = (r.^2.*m0.*expf1y0ind.*(r./(r + m0)).^r.*(expf1y0ind + (r./(r + m0)).^r + m0.*(r./(r + m0)).^r - expf1y0ind.*m0))./((r + m0).^2.*(expf1y0ind + (r./(r + m0)).^r).^3);
      % thrid derivative derivative wrt f1 (22)
      llg3(2,2,2,y0ind) = ((r.^2.*m0.*expf2y0ind.*(z0.*expf1y0ind2 + z0.*(r./(r + m0)).^(2.*r) + 3.*r.*z0.*expf1y0ind2) - r.^2.*m0.*(r.*(r./(r + m0)).^(2.*r) + r.*expf1y0ind2 + r.*expf1y0ind2.*m0.^2)).*(r./(r + m0)).^r + expf1y0ind.*(r.^2.*m0.*expf2y0ind.*(2.*z0 + 3.*r.*z0) - r.^2.*m0.*(2.*r - r.*m0.^2)).*(r./(r + m0)).^(2.*r))./((r + m0).^3.*(expf1y0ind + (r./(r + m0)).^r).^3);

      % y>0:
      llg3(1,1,1,yind) = -expf1(yind).*(1-expf1(yind))./(1+expf1(yind)).^3;
      llg3(2,2,2,yind) = - m(yind).*(r.^2 + y(yind).*r)./(r + m(yind)).^2 + 2.*m(yind).^2.*(r.^2 + y(yind).*r)./(r + m(yind)).^3;
      
      case 'latent2+param'
        
        % simplify(diff(-e/(1+e)^2 + e*((r/(r+m))^r)/(e+((r/(r+m))^r))^2,r))
        % simplify(diff((r/(r+m))^(r-1)*(m*r^2/(r+m)^2)*e/(e+((r/(r+m))^r))^2,r))
        % simplify(diff(-1./(e+((r./(r+m)).^r)).*( (r./(r+m)).^(2*r-2).*(m.*r.^2./(r+m).^2).^2./(e+((r./(r+m)).^r)) + (r-1).*(r./(r+m)).^(r-2).*(-m.^2.*r.^3./(r+m).^4) + (r./(r+m)).^(r-1).*(m.*r.^2.*(r+m)-2*m.^2.*r.^2)./(r+m).^3),r))
        
        llg3_11=zeros(n,1);
        llg3_12=zeros(n,1);
        llg3_22=zeros(n,1);          
         
        e0=expf1(y0ind);
        m0=m(y0ind);
        
        llg3_11(y0ind)=(e0.*(e0 - (r./(m0 + r)).^r).*(r./(m0 + r)).^r.*(m0 + m0.*log(r./(m0 + r)) + r.*log(r./(m0 + r))))./((e0 + (r./(m0 + r)).^r).^3.*(m0 + r));
        llg3_12(y0ind)=(e0.^2.*m0.*(r./(m0 + r)).^r.*(m0 + r.^2.*log(r./(m0 + r)) + m0.*r + m0.*r.*log(r./(m0 + r))) - e0.*m0.*(r./(m0 + r)).^(2.*r).*(r.^2.*log(r./(m0 + r)) - m0 + m0.*r + m0.*r.*log(r./(m0 + r))))./((e0 + (r./(m0 + r)).^r).^3.*(m0 + r).^2);
        llg3_22(y0ind)=-(m0.*r.*(r./(m0 + r)).^(2.*r).*(4.*e0.*m0 - 2.*e0.*m0.^2 + e0.*m0.^2.*r + e0.*r.^2.*log(r./(m0 + r)) + e0.*m0.*r + e0.*m0.*r.*log(r./(m0 + r)) + e0.*m0.*r.^2.*log(r./(m0 + r)) + e0.*m0.^2.*r.*log(r./(m0 + r))) + m0.*r.*(r./(m0 + r)).^r.*(2.*m0.*(r./(m0 + r)).^(2.*r) + 2*e0.^2.*m0 - 2.*e0.^2.*m0.^2 + e0.^2.*m0.*r - e0.^2.*m0.^2.*r + e0.^2.*r.^2.*log(r./(m0 + r)) - e0.^2.*m0.*r.^2.*log(r./(m0 + r)) - e0.^2.*m0.^2.*r.*log(r./(m0 + r)) + e0.^2.*m0.*r.*log(r./(m0 + r))))./((e0 + (r./(m0 + r)).^r).^3.*(m0 + r).^3);
        
        llg3_22(yind) = m(yind).*(y(yind).*r - 2.*r.*m(yind) - m(yind).*y(yind))./(r+m(yind)).^3;
        
        llg3 = [diag(llg3_11) diag(llg3_12); diag(llg3_12) diag(llg3_22)];
        
        % correction due to the log transformation
        llg3 = llg3.*lik.disper;
    end
  end

  function [lpyt,Ey, Vary] = lik_zinegbin_predy(lik, Ef, Covf, yt, zt)
  %LIK_ZINEGBIN_PREDY  Returns the predictive mean, variance and density of y
  %
  %  Description         
  %    [EY, VARY] = LIK_ZINEGBIN_PREDY(LIK, EF, VARF) takes a
  %    likelihood structure LIK, posterior mean EF and posterior
  %    covariance COVF of the latent variable and returns the
  %    posterior predictive mean EY and variance VARY of the
  %    observations related to the latent variables. This 
  %    subfunction is needed when computing posterior predictive 
  %    distributions for future observations.
  %        
  %    [Ey, Vary, PY] = LIK_ZINEGBIN_PREDY(LIK, EF, VARF YT, ZT)
  %    Returns also the predictive density of YT, that is 
  %        p(yt | zt) = \int p(yt | f, zt) p(f|y) df.
  %    This requires also the incedence counts YT, expected counts ZT.
  %    This subfunction is needed when computing posterior predictive 
  %    distributions for future observations.
  %
  %  See also
  %    GPLA_PRED, GPEP_PRED, GPMC_PRED

    if numel(zt)==0
      zt=ones(size(Ef));
    end

    ntest=size(zt,1);
    %avgE = zt;
    r = lik.disper;
    
    Py = zeros(size(zt));
    
    S=10000;
    for i1=1:ntest
      Sigm_tmp=Covf(i1:ntest:(2*ntest),i1:ntest:(2*ntest));
      Sigm_tmp=(Sigm_tmp+Sigm_tmp')./2;
      f_star=mvnrnd(Ef(i1:ntest:(2*ntest)), Sigm_tmp, S);
      
      m = exp(f_star(:,2)).*zt(i1);
      expf1=exp(f_star(:,1));
      
      if yt(i1)==0
        Py(i1)=mean(exp(-log(1+expf1) + log( expf1 + (r./(r+m)).^r )));
      else
        Py(i1)=mean(exp(-log(1+expf1) + r.*(log(r) - log(r+m)) + gammaln(r+yt(i1)) - gammaln(r) - gammaln(yt(i1)+1) + yt(i1).*(log(m) - log(r+m))));
      end
    end
    Ey = [];
    Vary = [];
    lpyt=log(Py);
  end

  function p = lik_zinegbin_invlink(lik, f, z)
  %LIK_ZINEGBIN_INVLINK  Returns values of inverse link function
  %             
  %  Description 
  %    P = LIK_ZINEGBIN_INVLINK(LIK, F) takes a likelihood structure LIK and
  %    latent values F and returns the values of inverse link function P.
  %    This subfunction is needed when using function gp_predprctmu.
  %
  %     See also
  %     LIK_ZINEGBIN_LL, LIK_ZINEGBIN_PREDY
  
    p = exp(f);
  end
  
  function reclik = lik_zinegbin_recappend(reclik, ri, lik)
  %RECAPPEND  Append the parameters to the record
  %
  %  Description 
  %    RECLIK = GPCF_ZINEGBIN_RECAPPEND(RECLIK, RI, LIK) takes a
  %    likelihood record structure RECLIK, record index RI and
  %    likelihood structure LIK with the current MCMC samples of
  %    the parameters. Returns RECLIK which contains all the old
  %    samples and the current samples from LIK. This subfunction
  %    is needed when using MCMC sampling (gp_mc).
  % 
  %  See also
  %    GP_MC

  % Initialize record
    if nargin == 2
      reclik.type = 'Zinegbin';
      reclik.nondiagW=true;

      % Initialize parameter
      reclik.disper = [];

      % Set the function handles
      reclik.fh.pak = @lik_zinegbin_pak;
      reclik.fh.unpak = @lik_zinegbin_unpak;
      reclik.fh.lp = @lik_zinegbin_lp;
      reclik.fh.lpg = @lik_zinegbin_lpg;
      reclik.fh.ll = @lik_zinegbin_ll;
      reclik.fh.llg = @lik_zinegbin_llg;    
      reclik.fh.llg2 = @lik_zinegbin_llg2;
      reclik.fh.llg3 = @lik_zinegbin_llg3;
      reclik.fh.predy = @lik_zinegbin_predy;
      reclik.fh.invlink = @lik_zinegbin_invlink;
      reclik.fh.recappend = @lik_zinegbin_recappend;
      reclik.p=[];
      reclik.p.disper=[];
      if ~isempty(ri.p.disper)
        reclik.p.disper = ri.p.disper;
      end
      return
    end
    
    reclik.disper(ri,:)=lik.disper;
    if ~isempty(lik.p.disper)
        reclik.p.disper = feval(lik.p.disper.fh.recappend, reclik.p.disper, ri, lik.p.disper);
    end
  end


