function [Eft, Covft, ljpyt] = gpep_jpred(gp, x, y, varargin)
%GPEP_PRED  Predictions with Gaussian Process EP approximation
%
%  Description
%    [EFT, COVFT] = GPEP_JPRED(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    predictive distribution at test inputs XT. Returns a posterior
%    mean EFT and covariance COVFT of latent variables.
%
%        Eft =  E[f | xt,x,y,th]  = K_fy*(Kyy+s^2I)^(-1)*y
%      Covft = Cov[f | xt,x,y,th] = K_fy - K_fy*(Kyy+s^2I)^(-1)*K_yf. 
%
%    Each row of X corresponds to one input vector and each row of
%    Y corresponds to one output vector.
%
%    [EFT, COVFT, LJPYT] = GPEP_JPRED(GP, X, Y, XT, 'yt', YT, ...) 
%    returns also logarithm of the predictive joint density JPYT of
%    the observations YT at test input locations XT. This can be
%    used for example in the cross-validation. Here Y has to be
%    vector.
%
%    [EF, COVF, LJPY] = GPEP_JPRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density PY of the training
%    observations Y.
%
%    OPTIONS is optional parameter-value pair
%      predcf - an index vector telling which covariance functions are 
%               used for prediction. Default is all (1:gpcfn). 
%               See additional information below.
%      tstind - a vector/cell array defining, which rows of X belong 
%               to which training block in *IC type sparse models. 
%               Default is []. In case of PIC, a cell array
%               containing index vectors specifying the blocking
%               structure for test data. IN FIC and CS+FIC a
%               vector of length n that points out the test inputs
%               that are also in the training set (if none, set
%               TSTIND = [])
%      yt     - optional observed yt in test points (see below)
%      z      - optional observed quantity in triplet (x_i,y_i,z_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, expected value 
%               for ith case. 
%      zt     - optional observed quantity in triplet (xt_i,yt_i,zt_i)
%               Some likelihoods may use this. For example, in case of 
%               Poisson likelihood we have z_i=E_i, that is, the expected 
%               value for the ith case. 
%
%    NOTE! In case of FIC and PIC sparse approximation the
%    prediction for only some PREDCF covariance functions is just
%    an approximation since the covariance functions are coupled in
%    the approximation and are not strictly speaking additive
%    anymore.
%
%    For example, if you use covariance such as K = K1 + K2 your
%    predictions Eft1 = gpep_pred(GP, X, Y, X, 'predcf', 1) and 
%    Eft2 = gpep_pred(gp, x, y, x, 'predcf', 2) should sum up to 
%    Eft = gpep_pred(gp, x, y, x). That is Eft = Eft1 + Eft2. With 
%    FULL model this is true but with FIC and PIC this is true only 
%    approximately. That is Eft \approx Eft1 + Eft2.
%
%    With CS+FIC the predictions are exact if the PREDCF covariance
%    functions are all in the FIC part or if they are CS
%    covariances.
%
%    NOTE! When making predictions with a subset of covariance
%    functions with FIC approximation the predictive variance can
%    in some cases be ill-behaved i.e. negative or unrealistically
%    small. This may happen because of the approximative nature of
%    the prediction.
%  
%  See also
%    GPEP_E, GPEP_G, GP_PRED, DEMO_SPATIAL, DEMO_CLASSIFIC
%
% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010      Heikki Peura
% Copyright (c) 2011-2012 Ville Tolvanen
% Copyright (c) 2012 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

  ip=inputParser;
  ip.FunctionName = 'GPEP_JPRED';
  ip=iparser(ip,'addRequired','gp', @isstruct);
  ip=iparser(ip,'addRequired','x', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addRequired','y', @(x) ~isempty(x) && isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addOptional','xt', [], @(x) isempty(x) || (isreal(x) && all(isfinite(x(:)))));
  ip=iparser(ip,'addParamValue','yt', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','z', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','zt', [], @(x) isreal(x) && all(isfinite(x(:))));
  ip=iparser(ip,'addParamValue','predcf', [], @(x) isempty(x) || ...
                   isvector(x) && isreal(x) && all(isfinite(x)&x>0));
  ip=iparser(ip,'addParamValue','tstind', [], @(x) isempty(x) || iscell(x) ||...
                   (isvector(x) && isreal(x) && all(isfinite(x)&x>0)));
  ip=iparser(ip,'addParamValue','fcorr', 'off', @(x) ismember(x, {'off', 'fact', 'cm2', 'on'}));
  if numel(varargin)==0 || isnumeric(varargin{1})
    % inputParser should handle this, but it doesn't
    ip=iparser(ip,'parse',gp, x, y, varargin{:});
  else
    ip=iparser(ip,'parse',gp, x, y, [], varargin{:});
  end
  xt=ip.Results.xt;
  yt=ip.Results.yt;
  z=ip.Results.z;
  zt=ip.Results.zt;
  predcf=ip.Results.predcf;
  tstind=ip.Results.tstind;
  fcorr=ip.Results.fcorr;
  if isempty(xt)
    xt=x;
    if isempty(yt)
      yt=y;
    end
    if isempty(zt)
      zt=z;
    end
  end

  if ~isequal(fcorr, 'off')
    warning('Marginal corrections not available for joint predictions');
    fcorr='off';
  end
  [tn, tnin] = size(x);
  
  switch gp.type
    % ============================================================
    % FULL
    % ============================================================
    case 'FULL'        % Predictions with FULL GP model
      %[e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);  
      [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);  
      
      [tautildee, nutildee, L, L2] = deal(p.tautilde, p.nutilde, p.L, p.La2);
      tautilde=tautildee(:,1);
      nutilde=nutildee(:,1);
      if isfield(gp.lik,'int_likparam') && gp.lik.int_likparam && ~gp.lik.inputparam
        % Give q(theta) to likelihood function to integrate ovet
        zt=[p.mf2 L2'*L2];
      end
      if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude && ~gp.lik.inputmagnitude
        zt=[zt p.mf3 p.La3'*p.La3];
      end
      if ~isfield(gp, 'lik_mono')
        if (isfield(gp.lik, 'int_likparam') && gp.lik.inputparam) || ...
            (isfield(gp.lik, 'int_magnitude') && gp.lik.inputmagnitude) ...
            || (isfield(gp.lik, 'int_likparam') && isfield(gp, 'comp_cf'))
          [K,C]=gp_trcov(gp,x,gp.comp_cf{1});
          kstarstar = gp_trcov(gp, xt, gp.comp_cf{1});
          K_nf=gp_cov(gp,xt,x,gp.comp_cf{1});
        else
          [K, C]=gp_trcov(gp,x);
          kstarstar = gp_trcov(gp, xt, predcf);
          K_nf=gp_cov(gp,xt,x,predcf);
        end
      else
        x2=x;
        y2=y;
        x=gp.xv;
        [K,C]=gp_dtrcov(gp,x2,x);
        kstarstar=gp_trcov(rmfield(gp,{'derivobs','lik_mono'}),xt);
        ntest=size(xt,1);
        K_nf=gp_dcov(gp,x2,xt,predcf)';
        K_nf(ntest+1:end,:)=[];
      end
%       [tautilde, nutilde, L] = deal(p.tautilde, p.nutilde, p.L);
      
%       [K, C]=gp_trcov(gp,x);
%       [kstarstar, C_nn] = gp_trcov(gp, xt, predcf);
      ntest=size(xt,1);
%       K_nf=gp_cov(gp,xt,x,predcf);
      [n,nin] = size(x);
      
      if size(tautildee,2)==1 && all(tautilde > 0) ...
          && ~(isequal(gp.latent_opt.optim_method, 'robust-EP') ...
               || isfield(gp, 'lik_mono'))
        % This is the usual case where likelihood is log concave
        % for example, Poisson and probit
        sqrttautilde = sqrt(tautilde);
        Stildesqroot = sparse(1:n, 1:n, sqrttautilde, n, n);
        
        if ~isfield(gp,'meanf')                
          if issparse(L)          % If compact support covariance functions are used 
                                  % the covariance matrix will be sparse
            z=Stildesqroot*ldlsolve(L,Stildesqroot*(C*nutilde));
          else
            z=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
          end

          Eft=K_nf*(nutilde-z);    % The mean, zero mean GP            
        else
          z = Stildesqroot*(L'\(L\(Stildesqroot*(C))));
          
          Eft_zm=K_nf*(nutilde-z*nutilde);              % The mean, zero mean GP    
          Ks = eye(size(z)) - z;                       % inv(K + S^-1)*S^-1                    
          Ksy = Ks*nutilde;
          [RB, RAR] = mean_jpredf(gp,x,xt,K_nf',Ks,Ksy,'EP',Stildesqroot.^2);
          
          Eft = Eft_zm + RB;        % The mean
        end
        

        % Compute variance
        if nargout > 1
          if issparse(L)
            V = ldlsolve(L, Stildesqroot*K_nf');
            Covft = kstarstar - K_nf*(Stildesqroot*V);
          else
            V = (L\Stildesqroot)*K_nf';
            Covft = kstarstar - V'*V;
          end
          if isfield(gp,'meanf')
            Covft = Covft + RAR;
          end
        end
      else
        % We might end up here if the likelihood is not log concave
        % For example Student-t likelihood.
        % NOTE! This does not work reliably yet
%         z=tautilde.*(L'*(L*nutilde));
%         Eft=K_nf*(nutilde-z);
%         
%         if nargout > 1
%           S = diag(tautilde);
%           V = K_nf*S*L';
%           Covft = kstarstar - (K_nf*S)*K_nf' + V*V';
%         end
        % An alternative implementation for avoiding negative variances
        [Eft,V]=pred_var(tautilde,K,K_nf,nutilde);
        Covft=kstarstar-V;
      end
      %         if nargout > 2
      %             Eyt = Eft;
      %             Varyt = Covft + (C_nn - kstarstar);
      %         end
      if isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam && gp.lik.inputparam
        tautilde=tautildee(:,2);
        nutilde=nutildee(:,2);
        [K, C]=gp_trcov(gp,x, gp.comp_cf{2});
        kstarstar = gp_trcov(gp, xt, gp.comp_cf{2});
        K_nf=gp_cov(gp,xt,x,gp.comp_cf{2});
        
        [Eft(:,2),V]=pred_var(tautilde,K,K_nf,nutilde);
        Covft(:,:,2)=kstarstar-V;
      end
      if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude && gp.lik.inputmagnitude
        tautilde=tautildee(:,end);
        nutilde=nutildee(:,end);
        [K, C]=gp_trcov(gp,x, gp.comp_cf{end});
        kstarstar = gp_trcov(gp, xt, gp.comp_cf{end});
        K_nf=gp_cov(gp,xt,x,gp.comp_cf{end});
        
        [Eft(:,end+1),V]=pred_var(tautilde,K,K_nf,nutilde);
        Covft(:,:,end+1)=kstarstar-V;
      end
      
      % ============================================================
      % FIC
      % ============================================================        
    case 'FIC'        % Predictions with FIC sparse approximation for GP
      %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [L, La, b] = deal(p.L, p.La2, p.b);

      % Here tstind = 1 if the prediction is made for the training set 
      if nargin > 6
        if ~isempty(tstind) && length(tstind) ~= size(x,1)
          error('tstind (if provided) has to be of same length as x.')
        end
      else
        tstind = [];
      end
      
      u = gp.X_u;
      m = size(u,1);
      
      K_fu = gp_cov(gp,x,u,predcf);          % f x u
      K_nu=gp_cov(gp,xt,u,predcf);
      K_uu = gp_trcov(gp,u,predcf);          % u x u, noiseless covariance K_uu
      K_uu = (K_uu+K_uu')./2;                % ensure the symmetry of K_uu

      kstarstar=gp_trvar(gp,xt,predcf);        

      % From this on evaluate the prediction
      % See Snelson and Ghahramani (2007) for details 
      %        p=iLaKfu*(A\(iLaKfu'*myytilde));
      p = b';
      
      ntest=size(xt,1);
      
      Eft = K_nu*(K_uu\(K_fu'*p));
      
      % if the prediction is made for training set, evaluate Lav also for prediction points
      if ~isempty(tstind)
        [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Kv_ff-Qv_ff;
        Eft(tstind) = Eft(tstind) + Lav.*p;
      end
      
      % Compute variance
      if nargout > 1
        %Covft(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
        Luu = chol(K_uu)';
        B=Luu\(K_fu');   
        B2=Luu\(K_nu');   
        Knf = B2'*B;                    
        Knn = B2'*B2 + diag(kstarstar - sum(B2.^2)');
        if ~isempty(tstind)
          Knf(tstind,:) = Knf(tstind,:) + diag(kstarstar(tstind) - sum(B.^2)');
        end
        
        Covft = Knn - Knf * ( diag(1./La) - L*L' ) * Knf';
      end
      % ============================================================
      % PIC
      % ============================================================
    case {'PIC' 'PIC_BLOCK'}        % Predictions with PIC sparse approximation for GP
                                    % Calculate some help matrices  
      u = gp.X_u;
      ind = gp.tr_index;
      %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [L, La, b] = deal(p.L, p.La2, p.b);
        
      
      K_fu = gp_cov(gp, x, u, predcf);           % f x u
      K_nu = gp_cov(gp, xt, u, predcf);          % n x u   
      K_uu = gp_trcov(gp, u, predcf);            % u x u, noiseles covariance K_uu
      K_uu = (K_uu+K_uu')./2;                    % ensure the symmetry of K_uu

      Luu = chol(K_uu)';
      B=Luu\(K_fu');
      B2 = Luu\(K_nu');


      % From this on evaluate the prediction
      % See Snelson and Ghahramani (2007) for details 
      %        p=iLaKfu*(A\(iLaKfu'*myytilde));
      p = b';

      iKuuKuf = K_uu\K_fu';
      
      w_bu=zeros(length(xt),length(u));
      w_n=zeros(length(xt),1);
      for i=1:length(ind)
        w_bu(tstind{i},:) = repmat((iKuuKuf(:,ind{i})*p(ind{i},:))', length(tstind{i}),1);
        K_nf = gp_cov(gp, xt(tstind{i},:), x(ind{i},:), predcf);              % n x u
        w_n(tstind{i},:) = K_nf*p(ind{i},:);
      end
      
      Eft = K_nu*(iKuuKuf*p) - sum(K_nu.*w_bu,2) + w_n;

      % Compute variance
      if nargout > 1
        % NOTE!
        % This is done with full matrices at the moment. 
        % Needs to be rewritten.
        Knn = B2'*B2;
        Knf = B2'*B;
        C = -L*L';
        for i=1:length(ind)
          La2 = gp_trcov(gp, xt(tstind{i},:), predcf) - B2(:,tstind{i})'*B2(:,tstind{i});
          Knn(ind{i},ind{i}) =  Knn(ind{i},ind{i}) + La2;
          Laa = gp_cov(gp, xt(tstind{i},:), x(ind{i},:),predcf) - B2(:,tstind{i})'*B(:,ind{i});
          Knf(tstind{i},ind{i}) =  Knf(tstind{i},ind{i}) + Laa;
          C(ind{i},ind{i}) = C(ind{i},ind{i}) + inv(La{i});
        end
        
        Covft = Knn - Knf * C * Knf';

      end
      % ============================================================
      % CS+FIC
      % ============================================================
    case 'CS+FIC'        % Predictions with CS+FIC sparse approximation for GP
                         % Here tstind = 1 if the prediction is made for the training set 
      if nargin > 6 
        if ~isempty(tstind) && length(tstind) ~= size(x,1)
          error('tstind (if provided) has to be of same length as x.')
        end
      else
        tstind = [];
      end
      
      u = gp.X_u;
      m = length(u);
      n = size(x,1);
      n2 = size(xt,1);
      
      %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [L, La, b] = deal(p.L, p.La2, p.b);

      % Indexes to all non-compact support and compact support covariances.
      cf1 = [];
      cf2 = [];
      % Indexes to non-CS and CS covariances, which are used for predictions
      predcf1 = [];
      predcf2 = [];    
      
      ncf = length(gp.cf);
      % Loop through all covariance functions
      for i = 1:ncf        
        % Non-CS covariances
        if ~isfield(gp.cf{i},'cs') 
          cf1 = [cf1 i];
          % If used for prediction
          if ~isempty(find(predcf==i))
            predcf1 = [predcf1 i]; 
          end
          % CS-covariances
        else
          cf2 = [cf2 i];           
          % If used for prediction
          if ~isempty(find(predcf==i))
            predcf2 = [predcf2 i]; 
          end
        end
      end
      if isempty(predcf1) && isempty(predcf2)
        predcf1 = cf1;
        predcf2 = cf2;
      end
      
      % Determine the types of the covariance functions used
      % in making the prediction.
      if ~isempty(predcf1) && isempty(predcf2)       % Only non-CS covariances
        ptype = 1;
        predcf2 = cf2;
      elseif isempty(predcf1) && ~isempty(predcf2)   % Only CS covariances
        ptype = 2;
        predcf1 = cf1;
      else                                           % Both non-CS and CS covariances
        ptype = 3;
      end
      
      K_fu = gp_cov(gp,x,u,predcf1);   % f x u
      K_uu = gp_trcov(gp,u,predcf1);   % u x u, noiseles covariance K_uu
      K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
      K_nu=gp_cov(gp,xt,u,predcf1);
      
      Kcs_nf = gp_cov(gp, xt, x, predcf2);
      Kcs_nn = gp_trcov(gp, xt, predcf2);
      
      p = b';
      ntest=size(xt,1);
      
      % Calculate the predictive mean according to the type of
      % covariance functions used for making the prediction
      if ptype == 1
        Eft = K_nu*(K_uu\(K_fu'*p));
      elseif ptype == 2
        Eft = Kcs_nf*p;
      else 
        Eft = K_nu*(K_uu\(K_fu'*p)) + Kcs_nf*p;        
      end

      % evaluate also Lav if the prediction is made for training set
      if ~isempty(tstind)
        [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf1);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Kv_ff-Qv_ff;
      end
      
      % Add also Lav if the prediction is made for training set
      % and non-CS covariance function is used for prediction
      if ~isempty(tstind) && (ptype == 1 || ptype == 3)
        Eft(tstind) = Eft(tstind) + Lav.*p;
      end

      % Evaluate the variance
      if nargout > 1
        
        Luu = chol(K_uu)';
        B=Luu\(K_fu');   
        B2=Luu\(K_nu');   

        Knf = B2'*B + Kcs_nf;
        k = gp_trvar(gp,xt,predcf1);
        Knn = B2'*B2 + diag(k - sum(B2.^2)') + Kcs_nn;
        if ~isempty(tstind)
          Knf(tstind,:) = Knf(tstind,:) + diag(k(tstind) - sum(B.^2)');
        end
        
        Covft = Knn - Knf * ( inv(La) - L*L' ) * Knf';

      end
      % ============================================================
      % DTC/(VAR)
      % ============================================================
    case {'DTC' 'VAR' 'SOR'}        % Predictions with DTC or variational sparse approximation for GP
      %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
      [L, La, b] = deal(p.L, p.La2, p.b);

      % Here tstind = 1 if the prediction is made for the training set 
      if nargin > 6
        if ~isempty(tstind) && length(tstind) ~= size(x,1)
          error('tstind (if provided) has to be of same length as x.')
        end
      else
        tstind = [];
      end
      
      u = gp.X_u;
      m = size(u,1);
      
      K_fu = gp_cov(gp,x,u,predcf);         % f x u
      K_nu=gp_cov(gp,xt,u,predcf);
      K_uu = gp_trcov(gp,u,predcf);          % u x u, noiseles covariance K_uu
      K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu

      kstarstar=gp_trvar(gp,xt,predcf);        

      % From this on evaluate the prediction
      p = b';
      
      ntest=size(xt,1);
      
      Eft = K_nu*(K_uu\(K_fu'*p));
      
      % if the prediction is made for training set, evaluate Lav also for prediction points
      if ~isempty(tstind)
        [Kv_ff, Cv_ff] = gp_trvar(gp, xt(tstind,:), predcf);
        Luu = chol(K_uu)';
        B=Luu\(K_fu');
        Qv_ff=sum(B.^2)';
        Lav = Kv_ff-Cv_ff;
        Eft(tstind) = Eft(tstind);% + Lav.*p;
      end
      
      if nargout > 1
        % Compute variances of predictions
        %Covft(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
        Luu = chol(K_uu)';
        B=Luu\(K_fu');   
        B2=Luu\(K_nu');   

        Covft = sum(B2'.*(B*(repmat(La,1,m).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
        switch gp.type
          case {'VAR' 'DTC'}
            Covft = kstarstar - Covft;
          case 'SOR'
            Covft = sum(B2.^2,1)' - Covft;
        end
      end
  end
  
  if nargout > 2
    [sampft] = gp_rnd(gp,x,y, xt, 'z', z, 'zt', zt, 'nsamp', 500);
    lpyt = zeros(500,1);
    for i=1:size(sampft,2)
      lpyt(i) = gp.lik.fh.ll(gp.lik, yt, sampft(:,i), z);
    end
    ljpyt = sumlogs(lpyt);
  end
  
  if nargout > 3
    error('Too many output arguments for GPEP_JPRED.')
  end
  

end

function [m,S]=pred_var(tau_q,K,A,b)

% helper function for determining
%
% m = A * inv( K+ inv(diag(tau_q)) ) * inv(diag(tau_q)) *b
% S = diag( A * inv( K+ inv(diag(tau_q)) ) * A)
%
% when the site variances tau_q may be negative
%

  ii1=find(tau_q>0); n1=length(ii1); W1=sqrt(tau_q(ii1));
  ii2=find(tau_q<0); n2=length(ii2); W2=sqrt(abs(tau_q(ii2)));

  m=A*b;
  b=K*b;
  S=zeros(size(A,1));
  u=0;
  U=0;
  if ~isempty(ii1)
    % Cholesky decomposition for the positive sites
    L1=(W1*W1').*K(ii1,ii1);
    L1(1:n1+1:end)=L1(1:n1+1:end)+1;
    L1=chol(L1);
    
    U = bsxfun(@times,A(:,ii1),W1')/L1;
    u = L1'\(W1.*b(ii1));
    
    m = m-U*u;
    S=S+U*U';
%     S = S+sum(U.^2,2);
  end

  if ~isempty(ii2)
    % Cholesky decomposition for the negative sites
    V=bsxfun(@times,K(ii2,ii1),W1')/L1;
    L2=(W2*W2').*(V*V'-K(ii2,ii2));
    L2(1:n2+1:end)=L2(1:n2+1:end)+1;
    
    [L2,pd]=chol(L2);
    if pd==0
      U = bsxfun(@times,A(:,ii2),W2')/L2 -U*(bsxfun(@times,V,W2)'/L2);
      u = L2'\(W2.*b(ii2)) -L2'\(bsxfun(@times,V,W2)*u);
      
      m = m+U*u;
      S = S-U*U';
%       S = S-sum(U.^2,2);
    else
      fprintf('Posterior covariance is negative definite.\n')
    end
  end

end
