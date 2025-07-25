function [Eft, Varft, lpyt, Eyt, Varyt] = gpep_pred(gp, x, y, varargin)
%GPEP_PRED  Predictions with Gaussian Process EP approximation
%
%  Description
%    [EFT, VARFT] = GPEP_PRED(GP, X, Y, XT, OPTIONS)
%    takes a GP structure together with matrix X of training
%    inputs and vector Y of training targets, and evaluates the
%    predictive distribution at test inputs XT. Returns a posterior
%    mean EFT and variance VARFT of latent variables.
%
%    [EFT, VARFT, LPYT] = GPEP_PRED(GP, X, Y, XT, 'yt', YT, OPTIONS)
%    returns also logarithm of the predictive density LPYT of the
%    observations YT at test input locations XT. This can be used
%    for example in the cross-validation. Here Y has to be a vector.
%
%    [EFT, VARFT, LPYT, EYT, VARYT] = GPEP_PRED(GP, X, Y, XT, OPTIONS)
%    returns also the posterior predictive mean EYT and variance VARYT.
%
%    [EF, VARF, LPY, EY, VARY] = GPEP_PRED(GP, X, Y, OPTIONS)
%    evaluates the predictive distribution at training inputs X
%    and logarithm of the predictive density LPY of the training
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
%      fcorr  - Method used for latent marginal posterior corrections. 
%               Default is 'off'. For EP possible method is 'fact'.
%               If method is 'on', 'fact' is used for EP.
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

% Copyright (c) 2007-2010 Jarno Vanhatalo
% Copyright (c) 2010      Heikki Peura
% Copyright (c) 2011      Pasi Jylänki
% Copyright (c) 2012 Aki Vehtari

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

ip=inputParser;
ip.FunctionName = 'GPEP_PRED';
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
ip=iparser(ip,'addParamValue','fcorr', 'off', @(x) ismember(x, {'off', 'fact', 'cm2', 'on','lr'}));
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
  if isempty(tstind)
    if iscell(gp)
      gptype=gp{1}.type;
    else
      gptype=gp.type;
    end
    switch gptype
      case {'FULL' 'VAR' 'DTC' 'SOR'}
        tstind = [];
      case {'FIC' 'CS+FIC'}
        tstind = 1:size(x,1);
      case 'PIC'
        if iscell(gp)
          tstind = gp{1}.tr_index;
        else
          tstind = gp.tr_index;
        end
    end
  end
  if isempty(yt)
    yt=y;
  end
  if isempty(zt)
    zt=z;
  end
end

  [tn, tnin] = size(x);
  [n, nout] = size(y);

  if isfield(gp.lik, 'nondiagW')
    switch gp.type
      % ============================================================
      % FULL
      % ============================================================
      case 'FULL'
        %[e, edata, eprior, tautilde, nutilde, BKnu, B, cholP, invPBKnu]= gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        [e, edata, eprior, p]= gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        if isnan(e)
            Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
            return
        end
        [nutilde, BKnu, B, cholP, invPBKnu]=deal(p.nutilde, p.BKnu, p.B, p.cholP, p.invPBKnu);
        
        if isfield(gp, 'comp_cf')  % own covariance for each ouput component
          multicf = true;
          if length(gp.comp_cf) ~= nout
            error('GPEP_PRED: the number of component vectors in gp.comp_cf must be the same as number of outputs.')
          end
          if ~isempty(predcf)
            if ~iscell(predcf) || length(predcf)~=nout
              error(['GPEP_PRED: if own covariance for each output component is used,'...
                     'predcf has to be cell array and contain nout (vector) elements.   '])
            end
          else
            predcf = gp.comp_cf;
          end
        else
          multicf = false;
          for i1=1:nout
            predcf2{i1} = predcf;
          end
          predcf=predcf2;
        end
        
        ntest=size(xt,1);
        % covariances between the training and test latents
        Kt = zeros(ntest,n,nout);
        if multicf
          for i1=1:nout
            Kt(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
          end
        else
          for i1=1:nout
            Kt(:,:,i1) = gp_cov(gp,xt,x,predcf{i1});
          end
        end
        
        % full ep with non-diagonal site covariances
        zz=zeros(n*nout,1);
        for k1=1:nout
          zz((1:n)+(k1-1)*n)=BKnu(:,k1)-B(:,:,k1)*invPBKnu;
        end
        
        
        %- posterior predictive mean
        Eft=zeros(ntest*nout,1);
        for z1=1:nout
          Eft((1:ntest)+(z1-1)*ntest)=Kt(:,:,z1)*(nutilde(:,z1)-zz((1:n)+(z1-1)*n));
        end
        
        if nargout > 1
          % posterior predictive covariance
          Covf=zeros(nout, nout, ntest);
          
          invcholPBKt=zeros(n,ntest,nout);
          for k1=1:nout
            invcholPBKt(:,:,k1)=cholP\(B(:,:,k1)*Kt(:,:,k1)');
          end
          
          %- update posterior covariance
          for k1=1:nout
            % covariances for the test latents
            kstarstar = gp_trvar(gp,xt,predcf{i1});
            
            Covf(k1,k1,:)=kstarstar-sum(Kt(:,:,k1)'.*(B(:,:,k1)*Kt(:,:,k1)'))'+sum(invcholPBKt(:,:,k1).*invcholPBKt(:,:,k1))';
            for j1=(k1+1):nout
              Covf(k1,j1,:)=sum(invcholPBKt(:,:,k1).*invcholPBKt(:,:,j1));
              Covf(j1,k1,:)=Covf(k1,j1,:);
            end
          end
          Varft=Covf;
          
        end
        
        % ============================================================
        % FIC
        % ============================================================
      case 'FIC'        % Predictions with FIC sparse approximation for GP
                        % ============================================================
                        % PIC
                        % ============================================================
      case {'PIC' 'PIC_BLOCK'}        % Predictions with PIC sparse approximation for GP
                                      % ============================================================
                                      % CS+FIC
                                      % ============================================================
      case 'CS+FIC'        % Predictions with CS+FIC sparse approximation for GP
    end
    
  else % isfield(gp.lik, 'nondiagW')
    switch gp.type
      % ============================================================
      % FULL
      % ============================================================
      case 'FULL'        % Predictions with FULL GP model
        %[e, edata, eprior, tautilde, nutilde, L] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        if isfield(gp.lik, 'int_likparam')
        
          [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
          if isnan(e)
              Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
              return
          end
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
          if (isfield(gp.lik, 'int_likparam') && gp.lik.inputparam) || ...
             (isfield(gp.lik, 'int_magnitude') && gp.lik.inputmagnitude) ...
              || (isfield(gp.lik, 'int_likparam') && isfield(gp, 'comp_cf'))
            [K,C]=gp_trcov(gp,x,gp.comp_cf{1});
            kstarstar = gp_trvar(gp, xt, gp.comp_cf{1});
            K_nf=gp_cov(gp,xt,x,gp.comp_cf{1});
          else
            [K, C]=gp_trcov(gp,x);
            kstarstar = gp_trvar(gp, xt, predcf);
            K_nf=gp_cov(gp,xt,x,predcf);
          end
          ntest=size(xt,1);
          [n,nin] = size(x);
        
          if size(tautildee,2)==1 && all(tautilde > 0) && ~isequal(gp.latent_opt.optim_method, 'robust-EP')
            % This is the usual case where likelihood is log concave
            % for example, Poisson and probit
            sqrttautilde = sqrt(tautilde(:,1));
            Stildesqroot = sparse(1:n, 1:n, sqrttautilde, n, n);
          
            if ~isfield(gp,'meanf')
              if issparse(L) % If compact support covariance functions are used
                             % the covariance matrix will be sparse
                zz=Stildesqroot*ldlsolve(L,Stildesqroot*(C*nutilde));
              else
                zz=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
              end
              Eft=K_nf*(nutilde-zz);    % The mean, zero mean GP
            else
              zz = Stildesqroot*(L'\(L\(Stildesqroot*(C))));
            
              Eft_zm=K_nf*(nutilde-zz*nutilde); % The mean, zero mean GP
              Ks = eye(size(zz)) - zz;           % inv(K + S^-1)*S^-1
              Ksy = Ks*nutilde;
              [RB RAR] = mean_predf(gp,x,xt,K_nf',Ks,Ksy,'EP',Stildesqroot.^2);
            
              Eft = Eft_zm + RB;            % The mean
            end
          
            % Compute variance
            if nargout > 1
              if issparse(L)
                V = ldlsolve(L, Stildesqroot*K_nf');
                Varft = kstarstar - sum(K_nf.*(Stildesqroot*V)',2);
              else
                V = (L\Stildesqroot)*K_nf';
                Varft = kstarstar - sum(V.^2)';
              end
              if isfield(gp,'meanf')
                Varft = Varft + RAR;
              end
            end
          else
            % We might end up here if the likelihood is not log concave
            % For example Student-t likelihood.
          
            %{
            zz=tautilde.*(L'*(L*nutilde));
            Eft=K_nf*(nutilde-zz);
            
            if nargout > 1
              S = diag(tautilde);
              V = K_nf*S*L';
              Varft = kstarstar - sum((K_nf*S).*K_nf,2) + sum(V.^2,2);
            end
            %}
            
            % An alternative implementation for avoiding negative variances
            [Eft,V]=pred_var(tautilde,K,K_nf,nutilde);
            Varft=kstarstar-V;
            
          end
          if isfield(gp.lik, 'int_likparam') && gp.lik.int_likparam && gp.lik.inputparam
            tautilde=tautildee(:,2);
            nutilde=nutildee(:,2);
            [K, C]=gp_trcov(gp,x, gp.comp_cf{2});
            kstarstar = gp_trvar(gp, xt, gp.comp_cf{2});
            K_nf=gp_cov(gp,xt,x,gp.comp_cf{2});
            
            [Eft(:,2),V]=pred_var(tautilde,K,K_nf,nutilde);
            Varft(:,2)=kstarstar-V;        
          end
          if isfield(gp.lik, 'int_magnitude') && gp.lik.int_magnitude && gp.lik.inputmagnitude
            tautilde=tautildee(:,end);
            nutilde=nutildee(:,end);
            [K, C]=gp_trcov(gp,x, gp.comp_cf{end});
            kstarstar = gp_trvar(gp, xt, gp.comp_cf{end});
            K_nf=gp_cov(gp,xt,x,gp.comp_cf{end});
            
            [Eft(:,end+1),V]=pred_var(tautilde,K,K_nf,nutilde);
            Varft(:,end+1)=kstarstar-V;        
          end
          
        else % isfield(gp.lik, 'int_likparam')
        
          [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
          if isnan(e)
              Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
              return
          end
          [tautilde, nutilde, L] = deal(p.tautilde, p.nutilde, p.L);
          
          if ~isfield(gp, 'lik_mono')
            [K, C]=gp_trcov(gp,x);
            kstarstar = gp_trvar(gp, xt, predcf);
            ntest=size(xt,1);
            K_nf=gp_cov(gp,xt,x,predcf);
            [n,nin] = size(x);
          else
            x2=x;
            y2=y;
            x=gp.xv;
            [K,C]=gp_dtrcov(gp,x2,x);
            kstarstar=gp_trvar(rmfield(gp,'derivobs'),xt);
            ntest=size(xt,1);
            K_nf=gp_dcov(gp,x2,xt,predcf)';
            K_nf(ntest+1:end,:)=[];
          end
        
          if all(tautilde > 0) ... 
                  && ~(isequal(gp.latent_opt.optim_method, 'robust-EP') ...
                       || isfield(gp, 'lik_mono'))
            % This is the usual case where likelihood is log concave
            % for example, Poisson and probit
            sqrttautilde = sqrt(tautilde);
            nstt=length(sqrttautilde);
            Stildesqroot = sparse(1:nstt, 1:nstt, sqrttautilde, nstt,  nstt);
            
            if ~isfield(gp,'meanf')
              if issparse(L)          % If compact support covariance functions a  re used
                                      % the covariance matrix will be sparse
                zz=Stildesqroot*ldlsolve(L,Stildesqroot*(C*nutilde));
              else
                zz=Stildesqroot*(L'\(L\(Stildesqroot*(C*nutilde))));
              end
              
              Eft=K_nf*(nutilde-zz);    % The mean, zero mean GP
            else
              zz = Stildesqroot*(L'\(L\(Stildesqroot*(C))));
              
              Eft_zm=K_nf*(nutilde-zz*nutilde); % The mean, zero mean GP
              Ks = eye(size(zz)) - zz;           % inv(K + S^-1)*S^-1
              Ksy = Ks*nutilde;
              [RB RAR] = mean_predf(gp,x,xt,K_nf',Ks,Ksy,'EP',Stildesqroot.^2);
              
              Eft = Eft_zm + RB;            % The mean
            end
            
            % Compute variance
            if nargout > 1
              if issparse(L)
                V = ldlsolve(L, Stildesqroot*K_nf');
                Varft = kstarstar - sum(K_nf.*(Stildesqroot*V)',2);
              else
                V = (L\Stildesqroot)*K_nf';
                Varft = kstarstar - sum(V.^2)';
              end
              if isfield(gp,'meanf')
                Varft = Varft + RAR;
              end
            end
          else
            % We might end up here if the likelihood is not log concave
            % For example Student-t likelihood.
            
            %{
            zz=tautilde.*(L'*(L*nutilde));
            Eft=K_nf*(nutilde-zz);
            
            if nargout > 1
              S = diag(tautilde);
              V = K_nf*S*L';
              Varft = kstarstar - sum((K_nf*S).*K_nf,2) + sum(V.^2,2);
            end
            %}
            
            % An alternative implementation for avoiding negative variances
            [Eft,V]=pred_var(tautilde,K,K_nf,nutilde);
            Varft=kstarstar-V;
            
            if nargout > 3 && isfield(gp, 'lik_mono') && isequal(gp.lik.type, 'Ga  ussian')
              % Gaussian likelihood with monotonicity -> analytical
              % predictions for f, see e.g. gp_monotonic, gpep_predgrad
              Eyt=Eft;
              Varyt=Varft+gp.lik.sigma2;
              if ~isempty(yt)
                lpyt=norm_lpdf(yt, Eyt, sqrt(Varyt));
              else
                lpyt=[];
              end
              return
            end
            
          end
        
        end % isfield(gp.lik, 'int_likparam')
        % ============================================================
        % FIC
        % ============================================================
      case 'FIC'        % Predictions with FIC sparse approximation for GP
                        %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        if isnan(e)
            Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
            return
        end
        [tautilde, nutilde, L, La, b] = deal(p.tautilde, p.nutilde, p.L, p.La2, p.b);
        
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
        
        if all(tautilde > 0) && ~isequal(gp.latent_opt.optim_method, 'robust-EP')
          
          % From this on evaluate the prediction
          % See Snelson and Ghahramani (2007) for details
          %        p=iLaKfu*(A\(iLaKfu'*mutilde));
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
            %Varft(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
            Luu = chol(K_uu)';
            B=Luu\(K_fu');
            B2=Luu\(K_nu');
            Varft = kstarstar - sum(B2'.*(B*(repmat(La,1,m).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
            
            % if the prediction is made for training set, evaluate Lav also for prediction points
            if ~isempty(tstind)
              Varft(tstind) = Varft(tstind) - 2.*sum( B2(:,tstind)'.*(repmat((La.\Lav),1,m).*B'),2) ...
                  + 2.*sum( B2(:,tstind)'*(B*L).*(repmat(Lav,1,m).*L), 2)  ...
                  - Lav./La.*Lav + sum((repmat(Lav,1,m).*L).^2,2);
            end
          end
          
        else
          % Robust-EP
          [Eft,V]=pred_var2(tautilde,nutilde,L,K_uu,K_fu,b,K_nu);
          Varft=kstarstar-V;
          
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
        if isnan(e)
            Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
            return
        end
        [L, La, b] = deal(p.L, p.La2, p.b);
        
        K_fu = gp_cov(gp, x, u, predcf);         % f x u
        K_nu = gp_cov(gp, xt, u, predcf);         % n x u
        K_uu = gp_trcov(gp, u, predcf);    % u x u, noiseles covariance K_uu
        
        % From this on evaluate the prediction
        % See Snelson and Ghahramani (2007) for details
        %        p=iLaKfu*(A\(iLaKfu'*mutilde));
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
          kstarstar = gp_trvar(gp, xt, predcf);
          KnfL = K_nu*(iKuuKuf*L);
          Varft = zeros(length(xt),1);
          for i=1:length(ind)
            v_n = gp_cov(gp, xt(tstind{i},:), x(ind{i},:), predcf);              % n x u
            v_bu = K_nu(tstind{i},:)*iKuuKuf(:,ind{i});
            KnfLa = K_nu*(iKuuKuf(:,ind{i})/chol(La{i}));
            KnfLa(tstind{i},:) = KnfLa(tstind{i},:) - (v_bu + v_n)/chol(La{i});
            Varft = Varft + sum((KnfLa).^2,2);
            KnfL(tstind{i},:) = KnfL(tstind{i},:) - v_bu*L(ind{i},:) + v_n*L(ind{i},:);
          end
          Varft = kstarstar - (Varft - sum((KnfL).^2,2));
          
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
        if isnan(e)
            Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
            return
        end
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
        K_uu = gp_trcov(gp,u,predcf1);     % u x u, noiseles covariance K_uu
        K_uu = (K_uu+K_uu')./2;     % ensure the symmetry of K_uu
        K_nu=gp_cov(gp,xt,u,predcf1);
        
        Kcs_nf = gp_cov(gp, xt, x, predcf2);
        
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
          Knn_v = gp_trvar(gp,xt,predcf);
          Luu = chol(K_uu)';
          B=Luu\(K_fu');
          B2=Luu\(K_nu');
          p = amd(La);
          iLaKfu = La\K_fu;
          % Calculate the predictive variance according to the type
          % covariance functions used for making the prediction
          if ptype == 1 || ptype == 3
            % FIC part of the covariance
            Varft = Knn_v - sum(B2'.*(B*(La\B')*B2)',2) + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
            % Add Lav2 if the prediction is made for the training set
            if  ~isempty(tstind)
              % Non-CS covariance
              if ptype == 1
                Kcs_nf = sparse(tstind,1:n,Lav,n2,n);
                % Non-CS and CS covariances
              else
                Kcs_nf = Kcs_nf + sparse(tstind,1:n,Lav,n2,n);
              end
              % Add Lav2 inside Kcs_nf
              Varft = Varft - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                      - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
              % In case of both non-CS and CS prediction covariances add
              % only Kcs_nf if the prediction is not done for the training set
            elseif ptype == 3
              Varft = Varft - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2) + sum((Kcs_nf*L).^2, 2) ...
                      - 2.*sum((Kcs_nf*iLaKfu).*(K_uu\K_nu')',2) + 2.*sum((Kcs_nf*L).*(L'*K_fu*(K_uu\K_nu'))' ,2);
            end
            % Prediction with only CS covariance
          elseif ptype == 2
            Varft = Knn_v - sum((Kcs_nf(:,p)/chol(La(p,p))).^2,2) + sum((Kcs_nf*L).^2, 2) ;
          end
        end
        % ============================================================
        % DTC/(VAR)
        % ============================================================
      case {'DTC' 'VAR' 'SOR'}        % Predictions with DTC or variational sparse approximation for GP
                                      %[e, edata, eprior, tautilde, nutilde, L, La, b] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        [e, edata, eprior, p] = gpep_e(gp_pak(gp), gp, x, y, 'z', z);
        if isnan(e)
            Eft=NaN; Varft=NaN; lpyt=NaN; Eyt=NaN; Varyt=NaN;
            return
        end
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
          %Varft(i1,1)=kstarstar(i1) - (sum(Knf(i1,:).^2./La') - sum((Knf(i1,:)*L).^2));
          Luu = chol(K_uu)';
          B=Luu\(K_fu');
          B2=Luu\(K_nu');
          
          Varft = sum(B2'.*(B*(repmat(La,1,m).\B')*B2)',2)  + sum((K_nu*(K_uu\(K_fu'*L))).^2, 2);
          switch gp.type
            case {'VAR' 'DTC'}
              Varft = kstarstar - Varft;
            case 'SOR'
              Varft = sum(B2.^2,1)' - Varft;
          end
        end
    end
  end
  if ~isequal(fcorr, 'off')
    % Do marginal corrections for samples
    [pc_predm, fvecm] = gp_predcm(gp, x, y, xt, 'z', z, 'ind', 1:size(xt,1), 'fcorr', fcorr);
    for i=1:size(xt,1)
      % Remove NaNs and zeros
      pc_pred=pc_predm(:,i);
      dii=isnan(pc_pred)|pc_pred==0;
      pc_pred(dii)=[];
      fvec=fvecm(:,i);
      fvec(dii)=[];
      % Compute mean correction
      Eft(i) = trapz(fvec.*(pc_pred./sum(pc_pred)));
    end
   end
  
  % ============================================================
  % Evaluate also the predictive mean and variance of new observation(s)
  % ============================================================    
  if ~isequal(fcorr, 'off')
    if nargout == 3
      if isempty(yt)
        lpyt=[];
      else
        lpyt = gp.lik.fh.predy(gp.lik, fvecm', pc_predm', yt, zt);
      end
    elseif nargout > 3
      [lpyt, Eyt, Varyt] = gp.lik.fh.predy(gp.lik, fvecm', pc_predm', yt, zt);
    end
  else
    if nargout == 3
      if isempty(yt)
        lpyt=[];
      else
        lpyt = gp.lik.fh.predy(gp.lik, Eft, Varft, yt, zt);
      end
    elseif nargout > 3
      [lpyt, Eyt, Varyt] = gp.lik.fh.predy(gp.lik, Eft, Varft, yt, zt);
    end
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
  S=zeros(size(A,1),1);
  u=0;
  U=0;
  L1=[];
  if ~isempty(ii1)
    % Cholesky decomposition for the positive sites
    L1=(W1*W1').*K(ii1,ii1);
    L1(1:n1+1:end)=L1(1:n1+1:end)+1;
    L1=chol(L1);
    
    U = bsxfun(@times,A(:,ii1),W1')/L1;
    u = L1'\(W1.*b(ii1));
    
    m = m-U*u;
    S = S+sum(U.^2,2);
  end

  if ~isempty(ii2)
    % Cholesky decomposition for the negative sites
    V=bsxfun(@times,K(ii2,ii1),W1')/L1;
    if isempty(V)
        V=0;
    end
    L2=(W2*W2').*(V*V'-K(ii2,ii2));
    L2(1:n2+1:end)=L2(1:n2+1:end)+1;
    
    [L2,pd]=chol(L2);
    if pd==0
      U = bsxfun(@minus, bsxfun(@times,A(:,ii2),W2')/L2,U*(bsxfun(@times,V,W2)'/L2));
      u = L2'\(W2.*b(ii2)) -L2'\(bsxfun(@times,V,W2)*u);
      
      m = m+U*u;
      S = S-sum(U.^2,2);
    else
      fprintf('Posterior covariance is negative definite.\n')
    end
  end

end

function [m_q,S_q]=pred_var2(tautilde,nutilde,L,K_uu,K_fu,D,K_nu)

% function for determining the parameters of the q-distribution
% when site variances tau_q may be negative
%
% q(f) = N(f|0,K)*exp( -0.5*f'*diag(tau_q)*f + nu_q'*f )/Z_q = N(f|m_q,S_q)
%
% S_q = inv(inv(K)+diag(tau_q)) where K is sparse approximation for prior
%       covariance
% m_q = S_q*nu_q;
%
% det(eye(n)+K*diag(tau_q))) = det(L1)^2 * det(L2)^2
% where L1 and L2 are upper triangular
%
% see Expectation consistent approximate inference (Opper & Winther, 2005)

  n=length(nutilde);

  U = K_fu;
  S = 1+tautilde.*D;
  B = tautilde./S;
  BUiL = bsxfun(@times, B, U)/L';
  % iKS = diag(B) - BUiL*BUiL';

  Ktnu = D.*nutilde + U*(K_uu\(U'*nutilde));
  m_q = nutilde - B.*Ktnu + BUiL*(BUiL'*Ktnu);
  kstar = K_nu*(K_uu\K_fu');
  m_q = kstar*m_q;

  S_q = sum(bsxfun(@times,B',kstar.^2),2) - sum((kstar*BUiL).^2,2);
  % S_q = kstar*iKS*kstar';


end
