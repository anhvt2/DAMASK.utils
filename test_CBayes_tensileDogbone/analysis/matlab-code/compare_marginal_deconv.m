% script to test the de-convolution-based approach
% and compare with consistent marginal approach in the 
% case of additive Gaussian noise and a Gaussian observed

% The model is nonlinear and the initial is uniform, so the
% initial prediction is non-Gaussian

rng(123)

% Samples of x (uniform initial)
N = 10000;
x = rand(N,2);
x(:,1) = 0.79+0.2*x(:,1);
x(:,2) = 1-4.5*sqrt(0.1) + 9*sqrt(0.1)*x(:,2);

% Samples of noise
M = 1; 
nvar = 0.0005;%0.0005;
eta = sqrt(nvar)*randn(N,M);

% Model:
q = eval2Dmodel(x,'nlinv');

figure
scatter(x(:,1),x(:,2),20,q,'filled')

% Observed density:
dmean = 0.275;
dvar = 0.001;
ddens = @(q) 1/sqrt(2*pi)/sqrt(dvar)*exp(-1/2*(q-dmean).^2/dvar);

% Consistent Marginal Approach
q_cm = q + eta;
pfinit_cm = km_kdeND(q_cm,[],q_cm);
data_cm = ddens(q_cm);
ratio_cm = data_cm./pfinit_cm;

% check that the updated density integrates to approx. 1
I_up_cm = mean(ratio_cm)

% estimate information gained (add 1e-13 to avoid NaNs)
KLdv_up_cm = mean(ratio_cm.*log(ratio_cm+1.0e-13))

% Use the ratio to perform rejection sampling
check = rand(N,1);
I_cm = find(ratio_cm./max(ratio_cm) >= check);
q_cm_keep = q_cm(I_cm,:);
x_cm_keep = x(I_cm,:);

% Check the mean and variance of the push-forward of the
mean_pf_up_cm = mean(q_cm_keep(:))
var_pf_up_cm = var(q_cm_keep(:))

% Plot the push-forwards and observed
qplot = linspace(min(q_cm),max(q_cm),100)';

pfinit = km_kdeND(q_cm,[],qplot);
pfup_cm = km_kdeND(q_cm_keep,[],qplot);
data = ddens(qplot);

figure
plot(qplot,data,'b','linewidth',2)
hold on
plot(qplot,pfinit,'k','linewidth',2)
plot(qplot,pfup_cm,'g','linewidth',2)
hold off
legend('Observed','PF Initial', 'PF Updated - CM')
set(gca,'FontSize',16)

figure
scatter(x(:,1),x(:,2),20,ratio_cm,'filled')
axis tight

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% De-convolve the noise from the observed density:
dmean_dc = 0.275 - 0.0; % noise is zero mean
dvar_dc = 0.001 - nvar;
ddens_dc = @(q) 1/sqrt(2*pi)/sqrt(dvar_dc)*exp(-1/2*(q-dmean_dc).^2/dvar_dc);

% Deconv Approach
pfinit_dc = km_kdeND(q,[],q);
data_dc = ddens_dc(q);
ratio_dc = data_dc./pfinit_dc;

% check that the updated/posterior density integrates to approx. 1
I_up_dc = mean(ratio_dc)

% estimate information gained (add 1e-13 to avoid NaNs)
KLdv_up_dc = mean(ratio_dc.*log(ratio_dc+1.0e-13))

% Use the ratio to perform rejection sampling
% check = rand(N,1);
I_dc = find(ratio_dc./max(ratio_dc) >= check);
q_dc_keep = q(I_dc,:) + eta(I_dc,:);
x_dc_keep = x(I_dc,:);

% Check the mean and variance of the push-forward of the
mean_pf_up_dc = mean(q_dc_keep(:))
var_pf_up_dc = var(q_dc_keep(:))

% Plot the push-forwards and observed
qplot = linspace(min(q_cm),max(q_cm),100)';

pfinit = km_kdeND(q+eta,[],qplot);
pfup_dc = km_kdeND(q_dc_keep,[],qplot);
data = ddens(qplot);

figure
plot(qplot,data,'b','linewidth',2)
hold on
plot(qplot,pfinit,'k','linewidth',2)
plot(qplot,pfup_dc,'g','linewidth',2)
hold off
legend('Observed','PF Initial', 'PF Updated - CM')
set(gca,'FontSize',16)

figure
scatter(x(:,1),x(:,2),20,ratio_dc,'filled')
axis tight

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The point here is that they are different answers to different questions
% Both are correct in their own way
