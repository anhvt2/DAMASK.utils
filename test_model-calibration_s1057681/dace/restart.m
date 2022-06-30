close all;
clear all;
home

% ------------------------------ input parameters ------------------------------
parentPath = 'D:\WEDA\Bayesian-optimization\impellerVaneDesignOpt\';
pumpAssem = 'Z0534'; % pump assembly number;
cd(parentPath); % change to parent path
addpath(strcat(parentPath,'dace')); % add GPR toolbox
addpath(parentPath); % add current path
d = 33; % dimensionality of the problem
scaledThetaVect(1)  = 0.5 * 1e3;
scaledThetaVect(2)  = 0.5 * 1e3;
scaledThetaVect(3)  = 0.5 * 1e3;
scaledThetaVect(4)  = 0.5 * 1e3;
scaledThetaVect(5)  = 0.5 * 1e3;
scaledThetaVect(6)  = 1.0 * 1e3;
scaledThetaVect(7)  = 1.0 * 1e3;
scaledThetaVect(8)  = 1.0 * 1e3;
scaledThetaVect(9)  = 1.0 * 1e3;
scaledThetaVect(10) = 1.0 * 1e3;
scaledThetaVect(11) = 8.0 * 1e3;
scaledThetaVect(12) = 8.0 * 1e3;
scaledThetaVect(13) = 8.0 * 1e3;
scaledThetaVect(14) = 8.0 * 1e3;
scaledThetaVect(15) = 8.0 * 1e3;
scaledThetaVect(16) = 8.0 * 1e3;
scaledThetaVect(17) = 8.0 * 1e3;
scaledThetaVect(18) = 8.0 * 1e3;
scaledThetaVect(19) = 8.0 * 1e3;
scaledThetaVect(20) = 100 * 1e3;
scaledThetaVect(21) = 100 * 1e3;
scaledThetaVect(22) = 100 * 1e3;
scaledThetaVect(23) = 8.0 * 1e3;
scaledThetaVect(24) = 8.0 * 1e3;
scaledThetaVect(25) = 8.0 * 1e3;
scaledThetaVect(26) = 8.0 * 1e3;
scaledThetaVect(27) = 8.0 * 1e3;
scaledThetaVect(28) = 8.0 * 1e3;
scaledThetaVect(29) = 8.0 * 1e3;
scaledThetaVect(30) = 8.0 * 1e3;
scaledThetaVect(31) = 8.0 * 1e3;
scaledThetaVect(32) = 8.0 * 1e3;
scaledThetaVect(33) = 8.0 * 1e3;
theta = scaledThetaVect * 200; lob = scaledThetaVect * 1e-1; upb = 20 * scaledThetaVect; % define init GPR params
maxiter = 2000; % maximum number of iterations
numInitPoint = 65; % number of initial random sampling point
fprintf('Running Bayesian-optimization on %s\n', pumpAssem);
m = 4; n = 2; % number of control points along and transverse the vane direction

% ------------------------------ initial guess ------------------------------

% initial guess -- read result from OLS to obtain Bezier control points
cppP = zeros(m + 1, n + 1, 3);
cppS = zeros(m + 1, n + 1, 3);

cd(parentPath);
S = []; Y = [];

for i = 1:numInitPoint
	currentDirectory = sprintf('%s_Iter%d',pumpAssem,i);
	cd(currentDirectory);

	cppP(:,:,1) = dlmread('cpp.pressure.z.txt');
	cppP(:,:,2) = dlmread('cpp.pressure.r.txt');
	cppP(:,:,3) = dlmread('cpp.pressure.theta.txt');
	cppS(:,:,1) = dlmread('cpp.suction.z.txt');
	cppS(:,:,2) = dlmread('cpp.suction.r.txt');
	cppS(:,:,3) = dlmread('cpp.suction.theta.txt');
	x = convertToSingleInput(cppP, cppS);
	% system('python36 getMaxWear.py'); % get max wear rate
	S = [S; x]; Y = [Y; dlmread('maxWear.txt')]; % feedback maxWear.txt into dmodel
	% [dmodel, perf] = dacefit(S,Y,@regpoly0,@corrgauss,theta,lob,upb); % update model
	cd(parentPath); % go back to the parent path
end

[dmodel, perf] = dacefit(S,Y,@regpoly0,@corrgauss,theta,lob,upb); % update model

% ------------------------------ optimization loop ------------------------------

for i = (numInitPoint + 1):maxiter
	system(sprintf('mkdir %s_Iter%d',pumpAssem,i));
	currentDirectory = sprintf('%s_Iter%d',pumpAssem,i);
	cd(currentDirectory);
	system(strcat('copy ..\',sprintf('%s',pumpAssem),'\* .')); % copy inputs from template folder
	system('del g*.dat'); % remove g*.dat file
	x = getNextSamplingPoint(dmodel, parentPath); % get new sampling points via acquisition function -- write outputs to file
	system('python36 writeGFile.py'); % gernate new gFile -- SOLVE z and FIX r
	system('..\Imp64bit.exe < dumInp.txt'); % run impeller code
	system('python36 getMaxWear.py'); % get max wear rate
	S = [S; x']; Y = [Y; dlmread('maxWear.txt')]; % feedback maxWear.txt into dmodel
	[dmodel, perf] = dacefit(S,Y,@regpoly0,@corrgauss,theta,lob,upb); % update model
	cd(parentPath); % go back to the parent path
end
