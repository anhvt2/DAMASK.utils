function gpcf = gpcf_exp(varargin)
%GPCF_EXP  Create an exponential covariance function
%
%  Description
%    GPCF = GPCF_EXP('PARAM1',VALUE1,'PARAM2,VALUE2,...) creates a
%    exponential covariance function structure in which the named
%    parameters have the specified values. Any unspecified
%    parameters are set to default values.
%
%    GPCF = GPCF_EXP(GPCF,'PARAM1',VALUE1,'PARAM2,VALUE2,...) 
%    modify a covariance function structure with the named
%    parameters altered with the specified values.
%  
%    Parameters for exponential covariance function [default]
%      magnSigma2        - magnitude (squared) [0.1]
%      lengthScale       - length scale for each input. [1]
%                          This can be either scalar corresponding
%                          to an isotropic function or vector
%                          defining own length-scale for each input
%                          direction.
%      magnSigma2_prior  - prior for magnSigma2  [prior_logunif]
%      lengthScale_prior - prior for lengthScale [prior_t]
%      metric            - metric structure used by the covariance function []
%      selectedVariables - vector defining which inputs are used [all]
%                          selectedVariables is shorthand for using
%                          metric_euclidean with corresponding components
%
%    Note! If the prior is 'prior_fixed' then the parameter in
%    question is considered fixed and it is not handled in
%    optimization, grid integration, MCMC etc.
%
%  See also
%    GP_SET, GPCF_*, PRIOR_*, METRIC_*
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010 Aki Vehtari
% Copyright (c) 2014 Arno Solin and Jukka Koskenranta

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

  if nargin>0 && ischar(varargin{1}) && ismember(varargin{1},{'init' 'set'})
    % remove init and set
    varargin(1)=[];
  end
  
  ip=inputParser;
  ip.FunctionName = 'GPCF_EXP';
  ip=iparser(ip,'addOptional','gpcf', [], @isstruct);
  ip=iparser(ip,'addParamValue','magnSigma2',0.1, @(x) isscalar(x) && x>0);
  ip=iparser(ip,'addParamValue','lengthScale',1, @(x) isvector(x) && all(x>0));
  ip=iparser(ip,'addParamValue','metric',[], @isstruct);
  ip=iparser(ip,'addParamValue','magnSigma2_prior', prior_logunif(), ...
                   @(x) isstruct(x) || isempty(x));
  ip=iparser(ip,'addParamValue','lengthScale_prior',prior_t(), ...
                   @(x) isstruct(x) || isempty(x));
  ip=iparser(ip,'addParamValue','selectedVariables',[], @(x) isempty(x) || ...
                   (isvector(x) && all(x>0)));
  ip=iparser(ip,'parse',varargin{:});
  gpcf=ip.Results.gpcf;
  
  if isempty(gpcf)
    init=true;
    gpcf.type = 'gpcf_exp';
  else
    if ~isfield(gpcf,'type') && ~isequal(gpcf.type,'gpcf_exp')
      error('First argument does not seem to be a valid covariance function structure')
    end
    init=false;
  end
  if init
    % Set the function handles to the subfunctions
    gpcf.fh.pak = @gpcf_exp_pak;
    gpcf.fh.unpak = @gpcf_exp_unpak;
    gpcf.fh.lp = @gpcf_exp_lp;
    gpcf.fh.lpg = @gpcf_exp_lpg;
    gpcf.fh.cfg = @gpcf_exp_cfg;
    gpcf.fh.ginput = @gpcf_exp_ginput;
    gpcf.fh.cov = @gpcf_exp_cov;
    gpcf.fh.trcov  = @gpcf_exp_trcov;
    gpcf.fh.trvar  = @gpcf_exp_trvar;
    gpcf.fh.recappend = @gpcf_exp_recappend;
    gpcf.fh.cf2ss = @gpcf_exp_cf2ss;
  end
  
  % Initialize parameters
  if init || ~ismember('lengthScale',ip.UsingDefaults)
    gpcf.lengthScale = ip.Results.lengthScale;
  end
  if init || ~ismember('magnSigma2',ip.UsingDefaults)
    gpcf.magnSigma2 = ip.Results.magnSigma2;
  end

  % Initialize prior structure
  if init
    gpcf.p=[];
  end
  if init || ~ismember('lengthScale_prior',ip.UsingDefaults)
    gpcf.p.lengthScale=ip.Results.lengthScale_prior;
  end
  if init || ~ismember('magnSigma2_prior',ip.UsingDefaults)
    gpcf.p.magnSigma2=ip.Results.magnSigma2_prior;
  end

  %Initialize metric
  if ~ismember('metric',ip.UsingDefaults)
    if ~isempty(ip.Results.metric)
      gpcf.metric = ip.Results.metric;
      gpcf = rmfield(gpcf, 'lengthScale');
      gpcf.p = rmfield(gpcf.p, 'lengthScale');
    elseif isfield(gpcf,'metric')
      if ~isfield(gpcf,'lengthScale')
        gpcf.lengthScale = gpcf.metric.lengthScale;
      end
      if ~isfield(gpcf.p,'lengthScale')
        gpcf.p.lengthScale = gpcf.metric.p.lengthScale;
      end
      gpcf = rmfield(gpcf, 'metric');
    end
  end
  
  % selectedVariables options implemented using metric_euclidean
  if ~ismember('selectedVariables',ip.UsingDefaults)
    if ~isfield(gpcf,'metric')
      if ~isempty(ip.Results.selectedVariables)
        gpcf.selectedVariables = ip.Results.selectedVariables;
%         gpcf.metric=metric_euclidean('components',...
%                                      num2cell(ip.Results.selectedVariables),...
%                                      'lengthScale',gpcf.lengthScale,...
%                                      'lengthScale_prior',gpcf.p.lengthScale);
%         gpcf = rmfield(gpcf, 'lengthScale');
%         gpcf.p = rmfield(gpcf.p, 'lengthScale');
      end
    elseif isfield(gpcf,'metric') 
      if ~isempty(ip.Results.selectedVariables)
        gpcf.metric=metric_euclidean(gpcf.metric,...
                                     'components',...
                                     num2cell(ip.Results.selectedVariables));
        if ~ismember('lengthScale',ip.UsingDefaults)
          gpcf.metric.lengthScale=ip.Results.lengthScale;
          gpcf = rmfield(gpcf, 'lengthScale');
        end
        if ~ismember('lengthScale_prior',ip.UsingDefaults)
          gpcf.metric.p.lengthScale=ip.Results.lengthScale_prior;
          gpcf.p = rmfield(gpcf.p, 'lengthScale');
        end
      else
        if ~isfield(gpcf,'lengthScale')
          gpcf.lengthScale = gpcf.metric.lengthScale;
        end
        if ~isfield(gpcf.p,'lengthScale')
          gpcf.p.lengthScale = gpcf.metric.p.lengthScale;
        end
        gpcf = rmfield(gpcf, 'metric');
      end
    end
  end
  
end

function [w,s,h] = gpcf_exp_pak(gpcf)
%GPCF_EXP_PAK  Combine GP covariance function parameters into
%              one vector
%
%  Description
%    W = GPCF_EXP_PAK(GPCF) takes a covariance function
%    structure GPCF and combines the covariance function
%    parameters and their hyperparameters into a single row
%    vector W. This is a mandatory subfunction used for 
%    example in energy and gradient computations.
%
%       w = [ log(gpcf.magnSigma2)
%             (hyperparameters of gpcf.magnSigma2) 
%             log(gpcf.lengthScale(:))
%             (hyperparameters of gpcf.lengthScale)]'
%
%  See also
%    GPCF_EXP_UNPAK

  w = []; s = {}; h=[];
  
  if ~isempty(gpcf.p.magnSigma2)
    w = [w log(gpcf.magnSigma2)];
    s = [s; 'log(exp.magnSigma2)'];
    h = [h 1];
    % Hyperparameters of magnSigma2
    [wh sh hh] = gpcf.p.magnSigma2.fh.pak(gpcf.p.magnSigma2);
    sh=strcat(repmat('prior-', size(sh,1),1),sh);
    w = [w wh];
    s = [s; sh];
    h = [h 1+hh];
  end        
  
  if isfield(gpcf,'metric')
    [wm sm hm] = gpcf.metric.fh.pak(gpcf.metric);
    w = [w wm];
    s = [s; sm];
    h = [h hm];
  else
    if ~isempty(gpcf.p.lengthScale)
      w = [w log(gpcf.lengthScale)];
      if numel(gpcf.lengthScale)>1
        s = [s; sprintf('log(exp.lengthScale x %d)',numel(gpcf.lengthScale))];
      else
        s = [s; 'log(exp.lengthScale)'];
      end
      h = [h ones(1, numel(gpcf.lengthScale))];
      % Hyperparameters of lengthScale
      [wh, sh, hh] = gpcf.p.lengthScale.fh.pak(gpcf.p.lengthScale);
      sh=strcat(repmat('prior-', size(sh,1),1),sh);
      w = [w wh];
      s = [s; sh];
      h = [h 1+hh];
    end
  end
  
end

function [gpcf, w] = gpcf_exp_unpak(gpcf, w)
%GPCF_EXP_UNPAK  Sets the covariance function parameters into
%                the structure
%
%  Description
%    [GPCF, W] = GPCF_EXP_UNPAK(GPCF, W) takes a covariance
%    function structure GPCF and a hyper-parameter vector W,
%    and returns a covariance function structure identical
%    to the input, except that the covariance hyper-parameters
%    have been set to the values in W. Deletes the values set to
%    GPCF from W and returns the modified W. This is a mandatory 
%    subfunction used for example in energy and gradient computations.
%
%    Assignment is inverse of  
%       w = [ log(gpcf.magnSigma2)
%             (hyperparameters of gpcf.magnSigma2)
%             log(gpcf.lengthScale(:))
%             (hyperparameters of gpcf.lengthScale)]'
%
%  See also
%    GPCF_EXP_PAK
  
  gpp=gpcf.p;
  if ~isempty(gpp.magnSigma2)
    gpcf.magnSigma2 = exp(w(1));
    w = w(2:end);
    % Hyperparameters of magnSigma2
    [p, w] = gpcf.p.magnSigma2.fh.unpak(gpcf.p.magnSigma2, w);
    gpcf.p.magnSigma2 = p;
  end
  
  if isfield(gpcf,'metric')
    [metric, w] = gpcf.metric.fh.unpak(gpcf.metric, w);
    gpcf.metric = metric;
  else            
    if ~isempty(gpp.lengthScale)
      i1=1;
      i2=length(gpcf.lengthScale);
      gpcf.lengthScale = exp(w(i1:i2));
      w = w(i2+1:end);
      % Hyperparameters of lengthScale
      [p, w] = gpcf.p.lengthScale.fh.unpak(gpcf.p.lengthScale, w);
      gpcf.p.lengthScale = p;
    end
  end
  
end

function lp = gpcf_exp_lp(gpcf)
%GPCF_EXP_LP  Evaluate the log prior of covariance function parameters
%
%  Description
%    LP = GPCF_EXP_LP(GPCF, X, T) takes a covariance function
%    structure GPCF and returns log(p(th)), where th collects the
%    parameters. This is a mandatory subfunction used for example 
%    in energy computations.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LPG, GP_E

% Evaluate the prior contribution to the error. The parameters that
% are sampled are transformed, e.g., W = log(w) where w is all
% the "real" samples. On the other hand errors are evaluated in
% the W-space so we need take into account also the Jacobian of
% transformation, e.g., W -> w = exp(W). See Gelman et al. (2013),
% Bayesian Data Analysis, third edition, p. 21.
  lp = 0;
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.magnSigma2)
    lp = lp +gpp.magnSigma2.fh.lp(gpcf.magnSigma2, ...
                   gpp.magnSigma2) +log(gpcf.magnSigma2);
  end
  
  if isfield(gpcf,'metric')
    lp = lp +gpcf.metric.fh.lp(gpcf.metric);
  elseif ~isempty(gpp.lengthScale)
    lp = lp +gpp.lengthScale.fh.lp(gpcf.lengthScale, ...
                   gpp.lengthScale) +sum(log(gpcf.lengthScale));
  end
end

function lpg = gpcf_exp_lpg(gpcf)
%GPCF_EXP_LPG  Evaluate gradient of the log prior with respect
%              to the parameters.
%
%  Description
%    LPG = GPCF_EXP_LPG(GPCF) takes a covariance function
%    structure GPCF and returns LPG = d log (p(th))/dth, where th
%    is the vector of parameters. This is a mandatory subfunction 
%    used for example in gradient computations.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LP, GP_G

  lpg = [];
  gpp=gpcf.p;
  
  if ~isempty(gpcf.p.magnSigma2)            
    lpgs = gpp.magnSigma2.fh.lpg(gpcf.magnSigma2, gpp.magnSigma2);
    lpg = [lpg lpgs(1).*gpcf.magnSigma2+1 lpgs(2:end)];
  end
  
  if isfield(gpcf,'metric')
    lpg_dist = gpcf.metric.fh.lpg(gpcf.metric);
    lpg=[lpg lpg_dist];
  else
    if ~isempty(gpcf.p.lengthScale)
      lll = length(gpcf.lengthScale);
      lpgs = gpp.lengthScale.fh.lpg(gpcf.lengthScale, gpp.lengthScale);
      lpg = [lpg lpgs(1:lll).*gpcf.lengthScale+1 lpgs(lll+1:end)];
    end
  end
  
end

function DKff = gpcf_exp_cfg(gpcf, x, x2, mask, i1)
%GPCF_EXP_CFG  Evaluate gradient of covariance function
%                 with respect to the parameters
%
%  Description
%    DKff = GPCF_EXP_CFG(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to th (cell array with matrix elements). This is a 
%    mandatory subfunction used for example in gradient computations.
%
%    DKff = GPCF_EXP_CFG(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_EXP_CFG(GPCF, X, [], MASK) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the diagonal of gradients of covariance matrix
%    Kff = k(X,X2) with respect to th (cell array with matrix
%    elements). This subfunction is needed when using sparse 
%    approximations (e.g. FIC).
%
%    DKff = GPCF_EXP_CFG(GPCF, X, X2, [], i) takes a covariance
%    function structure GPCF, a matrix X of input vectors and
%    returns DKff, the gradient of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    hyperparameter. This subfunction is needed when using memory
%    save option in gp_set.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LP, GP_G

  gpp=gpcf.p;

  i2=1;
  DKff = {};
  
  if nargin==5
    % Use memory save option
    savememory=1;
    if i1==0
      % Return number of hyperparameters
      i=0;
      if ~isempty(gpcf.p.magnSigma2)
        i=1;
      end
      if ~isempty(gpcf.p.lengthScale)
        i=i+length(gpcf.lengthScale);
      end
      DKff=i;
      return
    end
  else
    savememory=0;
  end

  % Evaluate: DKff{1} = d Kff / d magnSigma2
  %           DKff{2} = d Kff / d lengthScale
  % NOTE! Here we have already taken into account that the parameters
  % are transformed through log() and thus dK/dlog(p) = p * dK/dp
  % evaluate the gradient for training covariance
  if nargin == 2 || (isempty(x2) && isempty(mask))
    Cdm = gpcf_exp_trcov(gpcf, x);

    ii1=0;
    if ~isempty(gpcf.p.magnSigma2)
      ii1 = ii1 +1;
      DKff{ii1} = Cdm;
    end

    if isfield(gpcf,'metric')
      dist = gpcf.metric.fh.dist(gpcf.metric, x);
      distg = gpcf.metric.fh.distg(gpcf.metric, x);
      gprior_dist = gpcf.metric.fh.lpg(gpcf.metric);
      for i=1:length(distg)
        ii1 = ii1+1;
        DKff{ii1} = -Cdm.*distg{i};
      end
    else
      if isfield(gpcf,'selectedVariables');
        x = x(:,gpcf.selectedVariables);
      end
      [n, m] =size(x);
      if ~savememory
        i1=1:m;
      else
        if i1==1
          DKff=DKff{1};
          return
        end
        ii1=ii1-1;
        i1=i1-1;
      end
      if ~isempty(gpcf.p.lengthScale)
        % loop over all the lengthScales
        if length(gpcf.lengthScale) == 1
          % In the case of isotropic EXP (no ARD)
          s = 1./gpcf.lengthScale;
          dist = 0;
          for i=1:m
            dist = dist + (bsxfun(@minus,x(:,i),x(:,i)')).^2;
          end
          D = Cdm.*s.*sqrt(dist);
          ii1 = ii1+1;
          DKff{ii1} = D;
        else
          % In the case ARD is used
          s = 1./gpcf.lengthScale.^2;
          dist = 0;
          dist2 = 0;
          for i=1:m
            dist = dist + s(i).*(bsxfun(@minus,x(:,i),x(:,i)')).^2;
          end
          dist = sqrt(dist);
          for i=i1                     
            D = s(i).*Cdm.*(bsxfun(@minus,x(:,i),x(:,i)')).^2;
            D(dist~=0) = D(dist~=0)./dist(dist~=0);
            ii1 = ii1+1;
            DKff{ii1} = D;
          end
        end
      end
    end
    % Evaluate the gradient of non-symmetric covariance (e.g. K_fu)
  elseif nargin == 3 || isempty(mask)
    if size(x,2) ~= size(x2,2)
      error('gpcf_exp -> _ghyper: The number of columns in x and x2 has to be the same. ')
    end
    
    ii1=0;
    K = gpcf.fh.cov(gpcf, x, x2);
    if ~isempty(gpcf.p.magnSigma2)
      ii1 = ii1 +1;
      DKff{ii1} = K;
    end
    
    if isfield(gpcf,'metric')                
      dist = gpcf.metric.fh.dist(gpcf.metric, x, x2);
      distg = gpcf.metric.fh.distg(gpcf.metric, x, x2);
      gprior_dist = gpcf.metric.fh.lpg(gpcf.metric);
      for i=1:length(distg)
        ii1 = ii1+1;                    
        DKff{ii1} = -K.*distg{i};                    
      end
    else
      if isfield(gpcf,'selectedVariables')
        x = x(:,gpcf.selectedVariables);
        x2 = x2(:,gpcf.selectedVariables);
      end
      [n, m] =size(x);
      if ~savememory
        i1=1:m;
      else
        if i1==1
          DKff=DKff{1};
          return
        end
        ii1=ii1-1;
        i1=i1-1;
      end
      if ~isempty(gpcf.p.lengthScale)
        % Evaluate help matrix for calculations of derivatives with respect
        % to the lengthScale
        if length(gpcf.lengthScale) == 1
          % In the case of an isotropic EXP
          s = 1./gpcf.lengthScale;
          dist = 0;
          for i=1:m
            dist = dist + (bsxfun(@minus,x(:,i),x2(:,i)')).^2;
          end
          DK_l = s.*K.*sqrt(dist);
          ii1=ii1+1;
          DKff{ii1} = DK_l;
        else
          % In the case ARD is used
          s = 1./gpcf.lengthScale.^2;        % set the length
          dist = 0; 
          for i=1:m
            dist = dist + s(i).*(bsxfun(@minus,x(:,i),x2(:,i)')).^2;
          end
          dist = sqrt(dist);
          for i=i1
            D1 = s(i).*K.* bsxfun(@minus,x(:,i),x2(:,i)').^2;
            D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
            ii1=ii1+1;
            DKff{ii1} = D1;
          end
        end
      end
    end
    % Evaluate: DKff{1}    = d mask(Kff,I) / d magnSigma2
    %           DKff{2...} = d mask(Kff,I) / d lengthScale
  elseif nargin == 4 || nargin == 5
    ii1=0;
    
    if ~isempty(gpcf.p.magnSigma2) || (~savememory || all(i1==1))
      ii1 = ii1+1;
      DKff{ii1} = gpcf.fh.trvar(gpcf, x);   % d mask(Kff,I) / d magnSigma2
    end

    if isfield(gpcf,'metric')
      dist = 0;
      distg = gpcf.metric.fh.distg(gpcf.metric, x, [], 1);
      gprior_dist = gpcf.metric.fh.lpg(gpcf.metric);
      for i=1:length(distg)
        ii1 = ii1+1;
        DKff{ii1} = 0;
      end
    else
      if ~isempty(gpcf.p.lengthScale)
        for i2=1:length(gpcf.lengthScale)
          ii1 = ii1+1;
          DKff{ii1}  = 0; % d mask(Kff,I) / d lengthScale
        end
      end
    end
  end
  if savememory
    DKff=DKff{1};
  end
end

function DKff = gpcf_exp_ginput(gpcf, x, x2, i1)
%GPCF_EXP_GINPUT  Evaluate gradient of covariance function with 
%                 respect to x.
%
%  Description
%    DKff = GPCF_EXP_GINPUT(GPCF, X) takes a covariance function
%    structure GPCF, a matrix X of input vectors and returns
%    DKff, the gradients of covariance matrix Kff = k(X,X) with
%    respect to X (cell array with matrix elements). This subfunction
%    is needed when computing gradients with respect to inducing
%    inputs in sparse approximations.
%
%    DKff = GPCF_EXP_GINPUT(GPCF, X, X2) takes a covariance
%    function structure GPCF, a matrix X of input vectors
%    and returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2) with respect to X (cell array with matrix elements).
%    This subfunction is needed when computing gradients with 
%    respect to inducing inputs in sparse approximations.
%
%    DKff = GPCF_EXP_GINPUT(GPCF, X, X2, i) takes a covariance
%    function structure GPCF, a matrix X of input vectors
%    and returns DKff, the gradients of covariance matrix Kff =
%    k(X,X2), or k(X,X) if X2 is empty, with respect to ith 
%    covariate in X. This subfunction is needed when using 
%    memory save option in gp_set.
%
%  See also
%    GPCF_EXP_PAK, GPCF_EXP_UNPAK, GPCF_EXP_LP, GP_G
  
  [n, m] =size(x);
  ii1 = 0;
  
  if nargin==4
    % Use memory save option
    if i1==0
      % Return number of hyperparameters
      if isfield(gpcf,'selectedVariables')
        DKff=length(gpcf.selectedVariables);
      else
        DKff=m;
      end
      return
    end
  else
    i1=1:m;
  end
  
  if nargin == 2 || isempty(x2)
    K = gpcf.fh.trcov(gpcf, x);
    if isfield(gpcf,'metric')
      dist = gpcf.metric.fh.dist(gpcf.metric, x);
      gdist = gpcf.metric.fh.ginput(gpcf.metric, x);
      for i=1:length(gdist)
        ii1 = ii1+1;
        DKff{ii1} = -K.*gdist{ii1};
      end
    else
      if length(gpcf.lengthScale) == 1
        % In the case of an isotropic EXP
        s = repmat(1./gpcf.lengthScale.^2, 1, m);
      else
        s = 1./gpcf.lengthScale.^2;
      end
      dist=0;
      for i2=1:m
        dist = dist + s(i2).*(bsxfun(@minus,x(:,i2),x(:,i2)')).^2;
      end
      dist = sqrt(dist); 
      for j = 1:n
        for i=i1
          D1 = zeros(n,n);
          D1(j,:) = -s(i).*bsxfun(@minus,x(j,i),x(:,i)');
          D1 = D1 + D1';
          
          D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
          DK = D1.*K;
          ii1 = ii1 + 1;
          DKff{ii1} = DK;
        end
      end
    end
    
  elseif nargin == 3 || nargin == 4
    [n2, m2] =size(x2);
    K = gpcf.fh.cov(gpcf, x, x2);

    if isfield(gpcf,'metric')
      dist = gpcf.metric.fh.dist(gpcf.metric, x, x2);
      gdist = gpcf.metric.fh.ginput(gpcf.metric, x, x2);
      for i=1:length(gdist)
        ii1 = ii1+1;
        DKff{ii1}   = -K.*gdist{ii1};
      end
    else 
      if length(gpcf.lengthScale) == 1
        % In the case of an isotropic EXP
        s = repmat(1./gpcf.lengthScale.^2, 1, m);
      else
        s = 1./gpcf.lengthScale.^2;
      end
      
      dist=0;
      for i2=1:m
        dist = dist + s(i2).*(bsxfun(@minus,x(:,i2),x2(:,i2)')).^2;
      end
      dist = sqrt(dist); 
      for j = 1:n
        for i=i1
          D1 = zeros(n,n2);
          D1(j,:) = -s(i).*bsxfun(@minus,x(j,i),x2(:,i)');
          
          D1(dist~=0) = D1(dist~=0)./dist(dist~=0);
          DK = D1.*K;
          ii1 = ii1 + 1;
          DKff{ii1} = DK;
        end
      end
    end
  end
end

function C = gpcf_exp_cov(gpcf, x1, x2)
%GP_EXP_COV  Evaluate covariance matrix between two input vectors
%
%  Description        
%    C = GP_EXP_COV(GP, TX, X) takes in covariance function of a
%    Gaussian process GP and two matrixes TX and X that contain
%    input vectors to GP. Returns covariance matrix C. Every
%    element ij of C contains covariance between inputs i in TX
%    and j in X. This is a mandatory subfunction used for example in
%    prediction and energy computations.
%
%  See also
%    GPCF_EXP_TRCOV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV
  
  if isempty(x2)
    x2=x1;
  end

  if size(x1,2)~=size(x2,2)
    error('the number of columns of X1 and X2 has to be same')
  end

  if isfield(gpcf,'metric')
    dist = gpcf.metric.fh.dist(gpcf.metric, x1, x2);
    dist(dist<eps) = 0;
    C = gpcf.magnSigma2.*exp(-dist);
  else
    if isfield(gpcf,'selectedVariables')
      x1 = x1(:,gpcf.selectedVariables);
      x2 = x2(:,gpcf.selectedVariables);
    end
    [n1,m1]=size(x1);
    [n2,m2]=size(x2);
    C=zeros(n1,n2);
    ma2 = gpcf.magnSigma2;
    
    % Evaluate the covariance
    if ~isempty(gpcf.lengthScale)  
      s2 = 1./gpcf.lengthScale.^2;
      % If ARD is not used make s a vector of 
      % equal elements 
      if size(s2)==1
        s2 = repmat(s2,1,m1);
      end
      dist=zeros(n1,n2);
      for j=1:m1
        dist = dist + s2(j).*(bsxfun(@minus,x1(:,j),x2(:,j)')).^2;
      end
      C = ma2.*exp(-sqrt(dist));
    end
    C(C<eps)=0;
  end
end

function C = gpcf_exp_trcov(gpcf, x)
%GP_EXP_TRCOV  Evaluate training covariance matrix of inputs
%
%  Description
%    C = GP_EXP_TRCOV(GP, TX) takes in covariance function of a
%    Gaussian process GP and matrix TX that contains training
%    input vectors. Returns covariance matrix C. Every element ij
%    of C contains covariance between inputs i and j in TX.
%    This is a mandatory subfunction used for example in prediction 
%    and energy computations.
%
%
%  See also
%    GPCF_EXP_COV, GPCF_EXP_TRVAR, GP_COV, GP_TRCOV

  if isfield(gpcf,'metric')
    % If other than scaled euclidean metric
    dist = gpcf.metric.fh.dist(gpcf.metric, x);
    dist(dist<eps) = 0;
    C = gpcf.magnSigma2.*exp(-dist);
  else
    % If scaled euclidean metric
    % Try to use the C-implementation            
    C = trcov(gpcf, x);
%     C = NaN;
    if isnan(C)
      % If there wasn't C-implementation do here
      if isfield(gpcf, 'selectedVariables')
        x = x(:,gpcf.selectedVariables);
      end
      [n, m] =size(x);
      
      s = 1./(gpcf.lengthScale);
      s2 = s.^2;
      if size(s)==1
        s2 = repmat(s2,1,m);
      end
      ma2 = gpcf.magnSigma2;
      
      % Here we take advantage of the
      % symmetry of covariance matrix
      C=zeros(n,n);
      for i1=2:n
        i1n=(i1-1)*n;
        for i2=1:i1-1
          ii=i1+(i2-1)*n;
          for i3=1:m
            C(ii)=C(ii)+s2(i3).*(x(i1,i3)-x(i2,i3)).^2;       % the covariance function
          end
          C(i1n+i2)=C(ii);
        end
      end
      C = ma2.*exp(-sqrt(C));
      C(C<eps)=0;
    end
  end
end

function C = gpcf_exp_trvar(gpcf, x)
%GP_EXP_TRVAR  Evaluate training variance vector
%
%  Description
%    C = GP_EXP_TRVAR(GPCF, TX) takes in covariance function of a
%    Gaussian process GPCF and matrix TX that contains training
%    inputs. Returns variance vector C. Every element i of C
%    contains variance of input i in TX. This is a mandatory 
%    subfunction used for example in prediction and energy computations.
%
%  See also
%    GPCF_EXP_COV, GP_COV, GP_TRCOV

  [n, m] =size(x);

  C = ones(n,1).*gpcf.magnSigma2;
  C(C<eps)=0;
end

function reccf = gpcf_exp_recappend(reccf, ri, gpcf)
%RECAPPEND  Record append
%
%  Description
%    RECCF = GPCF_EXP_RECAPPEND(RECCF, RI, GPCF) takes a
%    covariance function record structure RECCF, record index RI
%    and covariance function structure GPCF with the current MCMC
%    samples of the parameters. Returns RECCF which contains
%    all the old samples and the current samples from GPCF.
%    This subfunction is needed when using MCMC sampling (gp_mc).
%
%  See also
%    GP_MC and GP_MC -> RECAPPEND

  if nargin == 2
    % Initialize the record
    reccf.type = 'gpcf_exp';

    % Initialize parameters
    reccf.lengthScale= [];
    reccf.magnSigma2 = [];

    % Set the function handles
    reccf.fh.pak = @gpcf_exp_pak;
    reccf.fh.unpak = @gpcf_exp_unpak;
    reccf.fh.lp = @gpcf_exp_lp;
    reccf.fh.lpg = @gpcf_exp_lpg;
    reccf.fh.cfg = @gpcf_exp_cfg;
    reccf.fh.cov = @gpcf_exp_cov;
    reccf.fh.trcov  = @gpcf_exp_trcov;
    reccf.fh.trvar  = @gpcf_exp_trvar;
    reccf.fh.recappend = @gpcf_exp_recappend;  
    reccf.p=[];
    reccf.p.lengthScale=[];
    reccf.p.magnSigma2=[];
    if isfield(ri.p,'lengthScale') && ~isempty(ri.p.lengthScale)
      reccf.p.lengthScale = ri.p.lengthScale;
    end
    if ~isempty(ri.p.magnSigma2)
      reccf.p.magnSigma2 = ri.p.magnSigma2;
    end
    if isfield(ri, 'selectedVariables')
        reccf.selectedVariables = ri.selectedVariables;
    end
  else
    % Append to the record
    gpp = gpcf.p;

    if ~isfield(gpcf,'metric')
      % record lengthScale
      reccf.lengthScale(ri,:)=gpcf.lengthScale;
      if isfield(gpp,'lengthScale') && ~isempty(gpp.lengthScale)
        reccf.p.lengthScale = gpp.lengthScale.fh.recappend(reccf.p.lengthScale, ri, gpcf.p.lengthScale);
      end
    end
    % record magnSigma2
    reccf.magnSigma2(ri,:)=gpcf.magnSigma2;
    if isfield(gpp,'magnSigma2') && ~isempty(gpp.magnSigma2)
      reccf.p.magnSigma2 = gpp.magnSigma2.fh.recappend(reccf.p.magnSigma2, ri, gpcf.p.magnSigma2);
    end
  end
end

function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = gpcf_exp_cf2ss(gpcf,x)
%GPCF_EXP_CF2SS Convert the covariance function to state space form
%
%  Description
%    Convert the covariance function to state space form such that
%    the process can be described by the stochastic differential equation
%    of the form: 
%      df(t)/dt = F f(t) + L w(t),
%    where w(t) is a white noise process. The observation model now 
%    corresponds to y_k = H f(t_k) + r_k, where r_k ~ N(0,sigma2).
%
%  References:
%    Simo Sarkka, Arno Solin, Jouni Hartikainen (2013).
%    Spatiotemporal learning via infinite-dimensional Bayesian
%    filtering and smoothing. IEEE Signal Processing Magazine,
%    30(4):51-61.
%

  % Check arguments
  if nargin < 2, x = []; end

  % Return model matrices, derivatives and parameter information
  [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = ...
      cf_exp_to_ss(gpcf.magnSigma2, gpcf.lengthScale);
  
  % Check which parameters are optimized
  if isempty(gpcf.p.magnSigma2), ind(1) = false; else ind(1) = true; end
  if isempty(gpcf.p.lengthScale), ind(2) = false; else ind(2) = true; end
  
  % Return only those derivatives that are needed
  dF    = dF(:,:,ind);
  dQc   = dQc(:,:,ind);
  dPinf = dPinf(:,:,ind);
  
end
