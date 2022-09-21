
% --------------------------------------- construct response GPR(s) ---------------------------------------
close all;
clear all;
home;
format longg;

% cd 'gpml-matlab-v4.2-2018-06-11'
addpath('gpml');
startup;
pkg load statistics;

% log: batch parallel implementation

% variables explanation:
% 	gpml: hallucinated gpml -- to be corrected at the end of each iteration
% 	gpml: real interpolation gpml (with hallucination in the infeasible regions)
%	hypClf: hallucinated gpml -- to be corrected at the end of each iteration
%	hypClf: classif GPR

% local files/variables
% 	{input,output,feasible,complete,batchID}.dat
% 
% global files/variables (correspondingly)
%	{S,Y,F,C,B}.dat
% 	input: parameterized input (row format)
%	output: objective functional value (scalar)
% 	feasible: feasibility of the design after simulation ran
%	batchID: what batch corresponds to the current folder? 0: initial; 1: acquisition; 2: explore; 3: exploreClf

% change: 
%	1. modelName, e.g. 
%	2. waitTime
%	3. queryX.sh: e.g. querySpk, queryFPGA
%			this file should be located in the parentPath
%		input:
%			input.dat
%		output: 
%			output.dat
%			feasible.dat
%			complete.dat
%			batchID.dat
%
% 	deprecated: 4. d -- dimensionality

% --------------------------------------- INSTRUCTION ---------------------------------------
% procedure to set up a GP-Hedge constrained asynchronously parallel BO applications
% Step 1: set up a post-processing script that return all the outputs
%			output.dat 
%			feasible.dat
%			complete.dat
%			batchID.dat
%			rewards.dat
% Step 2: create initial random samples with post-processed data
%			(deprecated -- now handle within mainprog.m): put 0 in the acquisitionScheme.dat in those initial samples 
% Step 3: create a queryX.sh, and replace in the mainprog.m file
%			this is an end-to-end from input.dat to {output,feasible,complete,batchID,rewards}.dat for each folder
%			so this queryX.sh must include the post-processing data as well
% Step 4: input file and format
%			(1) create a template at ${modelName}_Template
%			(2) format all the iteration as ${modelName}_Iter{1,2,3,...}
% Step 5: post-process: use buildLogs.sh to build globale {input,output,feasible,complete,batchID}.dat for post-processing analysis
%

% --------------------------------------- input parameters ---------------------------------------

%% simulation settings
modelName = 'SS304L'; % declare model name -- must match with "_Template/"
queryShellScript = 'sbatch.damask.solo'; % query Shell script -- end-to-end, from input.dat to {output,feasible,complete,batchID,rewards}.dat

%% define lower and upper bounds for the control variables
xLB = -0 * ones(1, 6);
xUB = +1 * ones(1, 6);

% add rough scale of inputs and outputs so that it can be centered around 1e0
xScale = 1e0 * ones(1,length(xLB));
yScale = [1e0];

d = length(xLB); % dimensionality of the problem

% %% define objective GPR params -- default
% % initial guess for theta hyper-parameters
% theta = 1 * ones(1,d);
% lob = 2.5e-1 * ones(1,d);
% upb = 7.5e+0 * ones(1,d);

% % feasibility classif-GPR
% thetaClf = 1e+0 * ones(1, d);
% lobClf = 1e-2 * ones(1, d);
% upbClf = 2e+1 * ones(1, d);

%% batch-size setting
exploitSize = 6;        % exploitation by hallucination in batch
exploreSize = 4;        % exploration by sampling at maximal mse
exploreClfSize = 0; % exploration by sampling at maximal for classif-GPR
batchSize = exploitSize + exploreSize + exploreClfSize; % total number of concurrent simulations

%% optimization settings
maxiter = 400; % maximum number of iterations
numInitPoint = 10; % last maximum number of iterations in the initial sampling phase
numParallelPoint = numInitPoint; % true for asynchornous batch-parallel % last maximum number of iterations in the batch parallel BO; constraint numParallelPoint >= numInitPoint; (cont)
% if no parallel for batch parallel BO, then numParallelPoint = numInitPoint

parentPath = pwd; % no / at the end
cd(parentPath); % change to parent path
addpath(parentPath); % add current path
addpath(strcat(parentPath,'/gpml')); % add GPR toolbox
addpath(strcat(parentPath,'/bayesOptSrc-gpml')); % add BO toolbox

fprintf('Initialization begun for Bayesian-optimization on %s\n', modelName);
checkTime = 0.10; % minutes to periodically check simulations if they are complete in MATLAB master optimization loop
waitTime = 4; % hours to stop waiting for a batch to finish; indeed run post-processing after this
% note: make sure $waitTime is consistent with other scripts during rolling simulations out, i.e. headers of qsub* in each simulation

% print batchSettings.dat for python3 script checker
batchFile = fopen('batchSettings.dat', 'w+');
fprintf(batchFile, '%d\n', exploitSize); % exploitSize
fprintf(batchFile, '%d\n', exploreSize); % exploreSize
fprintf(batchFile, '%d\n', exploreClfSize); % exploreClfSize
fclose(batchFile);

% print batchSettings.dat for python3 script checker
waitFile = fopen('waitTime.dat', 'w+');
fprintf(waitFile, '%.4f\n', waitTime); % waitTime
fclose(waitFile);

modelFile = fopen('modelName.dat', 'w+');
fprintf(modelFile, '%s\n', modelName);
fclose(modelFile);

fprintf('Initialization completed Bayesian-optimization on %s\n', modelName);

xLB = xLB ./ xScale; % rescale
xUB = xUB ./ xScale; % rescale

% --------------------------------------- initial sampling --------------------------------------- %

% rng(0);
% Sinit = [];
% for  i = 1:numInitPoint
% 	Sinit(i, :) = xLB + rand(1, d) * (xUB - xLB)';
% end
% dlmwrite('Sinit_examples.dat', Sinit, 'delimiter', ',', 'precision', '%0.16f');

% for i = 1:numInitPoint
% 	currentDirectory = sprintf('%s_Iter%d', modelName, i);
% 	cd(currentDirectory);
% 	x = Sinit(i,:); 
% 	x = reshape(x, 1, length(x));
% 	dlmwrite('input.dat', x .* xScale, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
% 	system(sprintf('cp ../%s_Template/* .', modelName));
% 	system(sprintf('bash ./%s', queryShellScript)); % end-to-end queryX.sh: from input.dat to {output,feasible,complete,batchID,rewards}.dat	
% 	cd(parentPath);
% end
% pause(300);

% --------------------------------------- read sampling --------------------------------------- %
S = []; Y = []; F = []; C = []; B = [];

cd(parentPath);
% read sequential sampling
for i = 1:numInitPoint
	% system(sprintf('rmdir /s /q %s_Iter%d',modelName,1));
	% system(sprintf('mkdir -p %s_Iter%d',modelName,1));
	currentDirectory = sprintf('%s_Iter%d', modelName, i);
	cd(currentDirectory);
	% local files
	x = dlmread('input.dat'); x = reshape(x, 1, length(x));
	y = dlmread('output.dat');
	f = dlmread('feasible.dat');
	% system('echo 1 > complete.dat'); % deprecated
	c = dlmread('complete.dat');
	%% deprecated: overwrite original files during restart
	% system('echo 0 > batchID.dat'); % deprecated
	batchID = dlmread('batchID.dat');
	% system('echo 0 > acquisitionScheme.dat'); % deprecated
	% global files
	S = [S; x./xScale]; Y = [Y; y./yScale]; F = [F; f]; C = [C; c]; B = [B; batchID];
	fprintf('done importing sequential initial iteration %d\n', i);
	cd(parentPath);
end

% % read parallel sampling -- WARM RESTART
% for i = (numInitPoint + 1):numParallelPoint
% 	folderList = dir(sprintf('%s_Iter%d*', modelName, i));
% 	for j = 1:length(folderList)
% 		currentDirectory = folderList(j).name;
% 		cd(currentDirectory);
% 		% local files
% 		x = dlmread('input.dat'); x = reshape(x, 1, length(x));
% 		y = dlmread('output.dat'); 
% 		f = dlmread('feasible.dat');
% 		system('echo 1 > complete.dat'); c = dlmread('complete.dat');
% 		batchID = dlmread('batchID.dat');
% 		% global files
% 		S = [S; x./xScale]; Y = [Y; y./yScale]; F = [F; f]; C = [C; c]; B = [B; batchID];
% 		fprintf('done importing parallel BO iteration %d folder %s\n', i, currentDirectory);
% 		cd(parentPath);
% 	end
% end

%% before interpolation
% dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16f');
% dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16f');
% dlmwrite('F.dat', F, 'delimiter', ',', 'precision', '%d');
% dlmwrite('C.dat', C, 'delimiter', ',', 'precision', '%d');
% dlmwrite('B.dat', B, 'delimiter', ',', 'precision', '%d');
l = length(Y);

% interpolate using feasible and hallucinate at infeasible regions
tic;
fprintf('\n\nFitting gpml...\n\n\n');

% covfunc = {@covMaterniso, 3}; ell = 1/4; sf = 1; % iso-Matern-3/2
% covfunc = {@covMaternard,5}; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]); % iso-Matern-5/2
gpIter = 5000;
covfunc = {@covSEard}; sf = 1 ; hyp.cov = log([rand(d, 1); sf]); % Gaussian with ARD
meanfunc = {@meanSum, {@meanLinear, @meanConst}}; hyp.mean = zeros(d+1, 1);
likfunc = @likGauss; sn = 1e-2; hyp.lik = log(sn);
hyp = minimize(hyp, @gp, -gpIter, @infGaussLik, meanfunc, covfunc, likfunc, S(F>0,:), Y(F>0));

for i = 1:length(Y)
	if F(i) == 0, Y(i) = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S(F>0,:), Y(F>0), S(i,:)); end
end

hyp = minimize(hyp, @gp, -gpIter, @infGaussLik, meanfunc, covfunc, likfunc, S, Y);

% [dmodelInterp, ~] = dacefit(S(F>0), Y(F>0), @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
% for i = 1:length(Y)
% 	if F(i) == 0, Y(i) = predictor(S(i,:), dmodelInterp); end
% end
% clear dmodelInterp;
% [gpml, ~] = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);

meanfuncClf = @meanConst; hypClf.mean = 0;
covfuncClf = @covSEard;   hypClf.cov = log(ones(d+1, 1));
likfuncClf = @likErf;
hypClf = minimize(hypClf, @gp, -gpIter, @infEP, meanfuncClf, covfuncClf, likfuncClf, S, F);
% [yp, s2p] = gp(hypClf, @infEP, meanfuncClf, covfuncClf, likfuncClf, S, F, S);

% [hypClf, ~] = dacefit(S, F, @regpoly0, @correxp, thetaClf, lobClf, upbClf, xLB, xUB);

%% after interpolation -- write to files; will be used in bayesOptSrc-gpml/getBatch.py
% rescale
for i = 1:d; S(:,i) = S(:,i) * xScale(i); end; Y = Y * yScale;

system('rm -fv S.dat Y.dat F.dat C.dat B.dat');
dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('F.dat', F, 'delimiter', '\n');
dlmwrite('C.dat', C, 'delimiter', '\n');
dlmwrite('B.dat', B, 'delimiter', '\n');
toc;
fprintf('Done fitting gpml...\n\n\n');

% rescale
for i = 1:d; S(:,i) = S(:,i) / xScale(i); end; Y = Y / yScale;


% --------------------------------------- parallel optimization loop --------------------------------------- %


for i = (numParallelPoint + 1):maxiter


	%% write banner
	fprintf('\n\n---------------------------------------------------------\n');
	fprintf('\nRunning %d concurrent simulations.\n', batchSize);
	fprintf('Exploitation batch size: %d\n', exploitSize);
	fprintf('Exploration batch size: %d\n', exploreSize);
	fprintf('Exploration classification batch size: %d\n', exploreClfSize);
	fprintf('Iteration: %d\n', i);
	fprintf('\n---------------------------------------------------------\n\n\n');

	cd(parentPath); 
	system('python3 bayesOptSrc-gpml/checkComplete.py');
	system('python3 bayesOptSrc-gpml/getBatch.py'); % return the batchID for next query; dump to batchID.dat
	batchID = dlmread('batchID.dat');


	%% if there is no free batch then wait
	while batchID == 4
		system('rm -v batchID.dat');
		system('python3 bayesOptSrc-gpml/checkComplete.py');
		system('python3 bayesOptSrc-gpml/getBatch.py'); % return the batchID for next query; dump to batchID.dat
		batchID = dlmread('batchID.dat');
		pause(checkTime * 60);
	end

	currentFolder = sprintf('%s_Iter%d', modelName, i);
	system(sprintf('mkdir -p %s', currentFolder)); % system([fprintf('cp -rfv %s_Template %s', modelName, currentFolder)]);
	cd(currentFolder);
	system(sprintf('cp ../%s_Template/* .', modelName));
	system('mv -v ../batchID.dat .'); % output of getBatch.py


	%% sample acquisition scheme based on rewards
	if batchID == 1
		system('python3 ../bayesOptSrc-gpml/getRewards.py'); % return the rewards for sampling acquisition; dump to R.dat
		system('python3 ../bayesOptSrc-gpml/getAcquisitionScheme.py'); % dump to acquisitionScheme.dat
		acquisitionFunction = fileread('acquisitionScheme.dat');
		x = getNextSamplingPoint(hyp, meanfunc, covfunc, likfunc, hypClf, meanfuncClf, covfuncClf, likfuncClf, acquisitionFunction, xLB, xUB, S, Y, F); % batch acquisition
	elseif batchID == 2
		x = getNextSamplingPointReduceMSE(hyp, meanfunc, covfunc, likfunc, hypClf, meanfuncClf, covfuncClf, likfuncClf, acquisitionFunction, xLB, xUB, S, Y, F); % batch explore
	elseif batchID == 3
		x = getNextSamplingPointClfReduceMSE(hyp, meanfunc, covfunc, likfunc, hypClf, meanfuncClf, covfuncClf, likfuncClf, acquisitionFunction, xLB, xUB, S, Y, F); % batch exploreClf
	elseif batchID == 4
		fprintf('mainprog.m: getBatch.py: All batches are full\n');
	else
		fprintf('mainprog.m: Error: cannot identify batch\n');
	end

	x = reshape(x, 1, length(x));
	dlmwrite('input.dat', x .* xScale, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh


	%% write predictions to gpPredictions.dat
	% [mu, ~, rmse, ~] = predictor(x, gpml);
	[mu, rmse] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S, Y, x);
	predictFile = fopen('gpPredictions.dat', 'w+');
	fprintf(predictFile, '%0.8f\n%0.8f\n', mu * yScale, rmse * yScale^2);
	fclose(predictFile);


	%% query functional evaluation
	system('echo 0 > complete.dat'); % echo 0 > complete.dat; indicate case has been queried
	% system(sprintf('bash ./%s', queryShellScript)); % end-to-end queryX.sh: from input.dat to {output,feasible,complete,batchID,rewards}.dat
	system(sprintf('ssubmit')); % submit in SLURM scheduler system with convenience
	% note: system() returns MKL errors
	cd(parentPath);
	system('python3 bayesOptSrc-gpml/updateDb.py'); % write to {S,Y,F,C}.dat; no B.dat


	%% update inputs and outputs from {input,output,feasible,compelte}.dat at each folder
	S = dlmread('S.dat');
	Y = dlmread('Y.dat');
	F = dlmread('F.dat');
	C = dlmread('C.dat');


	%% rescale inputs and outputs
	% resize
	[~, dp] = size(S); if dp ~= d; S = S'; end;
	% rescale
	for i = 1:d; S(:,i) = S(:,i) / xScale(i); end; Y = Y / yScale;


	%% hallucinate and fit gpml
	fprintf('\n\nFitting gpml...\n\n\n');
	% theta = gpml.theta; % initialize from the previous hyper-parameters
	% [dmodelInterp, ~] = dacefit(S(F>0,:), Y(F>0), @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
	hyp = minimize(hyp, @gp, -gpIter, @infGaussLik, meanfunc, covfunc, likfunc, S(F>0,:), Y(F>0));

	for i = 1:length(Y)
		% if F(i) == 0, Y(i) = predictor(S(i,:), dmodelInterp); end
		if F(i) == 0, Y(i) = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, S(F>0,:), Y(F>0), S(i,:)); end
	end
	hyp = minimize(hyp, @gp, -gpIter, @infGaussLik, meanfunc, covfunc, likfunc, S, Y);
	
	% theta = dmodelInterp.theta; % initialize from the previous hyper-parameters
	% clear dmodelInterp;
	% [gpml, ~] = dacefit(S, Y, @regpoly2, @corrgauss, theta, lob, upb, xLB, xUB);
	% thetaClf = hypClf.theta; % initialize from the previous hyper-parameters
	% [hypClf, ~] = dacefit(S, F, @regpoly0, @correxp, thetaClf, lobClf, upbClf, xLB, xUB);

	hypClf = minimize(hypClf, @gp, -gpIter, @infEP, meanfuncClf, covfuncClf, likfuncClf, S, 2*F-1);
	pause(0.25);

end


exit;

