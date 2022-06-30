
% ------------------------------------------------------------ construct response GPR(s) ------------------------------------------------------------
close all;
clear all;
home;
format longg;
addpath('GPstuff-4.7');

% log: batch parallel implementation

% local files/variables
% 	{input,output,feasible,complete,batchID}.dat
% 
% global files/variables (correspondingly)
%	{S,Y,F,C,B}.dat
% 	input: parameterized input (row format)
%	output: objective functional value (scalar)
% 	feasible: feasibility of the design after simulation ran
%	batchID: what batch corresponds to the current folder? 0: initial; 1: acquisition; 2: explore; 3: exploreClassif

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

% ------------------------------------------------------------ INSTRUCTION ------------------------------------------------------------
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

% ------------------------------------------------------------ input parameters ------------------------------------------------------------

%% simulation settings
modelName = 'Zdt6'; % declare model name -- must match with "_Template/"
queryShellScript = 'queryGPStuff.sh'; % query Shell script -- end-to-end, from input.dat to {output,feasible,complete,batchID,rewards}.dat

%% define lower and upper bounds for the control variables
d = 10;					% reduced dimensionality of the problem
D = 10000; 				% true dimensionality of the problem
zLB = (-sqrt(d)) * ones(1,d);	% projected bounds of input
zUB = (+sqrt(d)) * ones(1,d);	% projected bounds of input
xLB = - ones(1,D);				% true bounds of input
xUB = + ones(1,D);				% true bounds of input

maxInducingPoints = 400; % max number of inducing points: length(X_u) = min(maxInducingPoints, length(x)); 

%% batch-size setting
exploitSize = 1;        % exploitation by hallucination in batch
exploreSize = 0;        % exploration by sampling at maximal mse
exploreClassifSize = 0; % exploration by sampling at maximal for classif-GPR
batchSize = exploitSize + exploreSize + exploreClassifSize; % total number of concurrent simulations

%% optimization settings
maxiter = 200; 			% maximum number of iterations
numInitPoint = d; 		% last maximum number of iterations in the initial sampling phase
numParallelPoint = numInitPoint; % true for asynchornous batch-parallel % last maximum number of iterations in the batch parallel BO; constraint numParallelPoint >= numInitPoint; (cont)
% if no parallel for batch parallel BO, then numParallelPoint = numInitPoint

parentPath = pwd; 								% no "/" at the end
cd(parentPath); 								% change to parent path
addpath(parentPath); 							% add current path
addpath(strcat(parentPath,'/bayesOptSrc')); 	% add BO toolbox
startup; 										% for GPStuff

fprintf('Initialization begun for Bayesian-optimization on %s\n', modelName);
checkTime = 0.10; 		% minutes to periodically check simulations if they are complete in MATLAB master optimization loop
waitTime = 1; 			% hours to stop waiting for a batch to finish; indeed run post-processing after this
% note: make sure $waitTime is consistent with other scripts during rolling simulations out, i.e. headers of qsub* in each simulation

% print batchSettings.dat for python3 script checker
batchFile = fopen('batchSettings.dat', 'w+');
fprintf(batchFile, '%d\n', exploitSize); % exploitSize
fprintf(batchFile, '%d\n', exploreSize); % exploreSize
fprintf(batchFile, '%d\n', exploreClassifSize); % exploreClassifSize
fclose(batchFile);

% print batchSettings.dat for python3 script checker
waitFile = fopen('waitTime.dat', 'w+');
fprintf(waitFile, '%.4f\n', waitTime); % waitTime
fclose(waitFile);

modelFile = fopen('modelName.dat', 'w+');
fprintf(modelFile, '%s\n', modelName);
fclose(modelFile);

fprintf('Initialization completed Bayesian-optimization on %s\n', modelName);

% xLB = xLB ./ xScale; % rescale
% xUB = xUB ./ xScale; % rescale

%% random embeddings settings
% A = randn(D, d);  			% a random matrix
A = eye(D, D); 			% default: no random embedding

%% GPStuff settings
pl = prior_t('s2', 1);
pm = prior_logunif();
pn = prior_logunif();

lik = lik_gaussian('sigma2', 0.2^2, 'sigma2_prior', pn);
gpcf = gpcf_matern32();
opt = optimset('TolFun', 1e-4, 'TolX', 1e-4);

% gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf, 'X_u', Z_u, 'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood');
% gp_fic = gp_optim(gp_fic, x, y, 'opt', opt);

% ------------------------------------------------------------ initial sampling ------------------------------------------------------------ %
% 
% rng(default); 	% for reproducibility/debug purposes
Sinit = []; Yinit = [];
for  i = 1:numInitPoint
	z = zLB + rand(1, d) .* (zUB - zLB);
	x = projectEmbedding(A, z, zLB, zUB, xLB, xUB);
	% [~, y] = feval(modelName, x);
	currentFolder = sprintf('%s_Iter%d', modelName, i);
	system(sprintf('mkdir -p %s', currentFolder)); % system([fprintf('cp -rfv %s_Template %s', modelName, currentFolder)]);
	cd(currentFolder);
	system(sprintf('cp ../%s_Template/* .', modelName));
	dlmwrite('input.dat', x, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
	dlmwrite('projectedInput.dat', z, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
	system('echo 0 > batchID.dat');
	system(sprintf('bash ./%s', queryShellScript)); % end-to-end queryX.sh: from input.dat to {output,feasible,complete,batchID,rewards}.dat
	pause(2);
	y = dlmread('output.dat');
	Sinit(i, :) = z;
	Yinit(i,:) = y; % minimization
	% currentDirectory = sprintf('%s_Iter%d', modelName, i);
	% cd(currentDirectory);
	% system(sprintf('mkdir -p %s_Iter%d', currentDirectory));
	% dlmwrite('input.dat', x, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
	% system(sprintf('cp ../%s_Template/* .', modelName));
	% system(sprintf('bash ./%s', queryShellScript)); % end-to-end queryX.sh: from input.dat to {output,feasible,complete,batchID,rewards}.dat	
	cd(parentPath);
end
% dlmwrite('Sinit_examples.dat', Sinit, 'delimiter', ',', 'precision', '%0.16f');
% 
% for i = 1:numInitPoint
% 	% currentDirectory = sprintf('%s_Iter%d', modelName, i);
% 	% cd(currentDirectory);
% 	x = Sinit(i,:); 
% 	x = reshape(x, 1, length(x));
% 	dlmwrite('input.dat', x .* xScale, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
% 	% system(sprintf('cp ../%s_Template/* .', modelName));
% 	% system(sprintf('bash ./%s', queryShellScript)); % end-to-end queryX.sh: from input.dat to {output,feasible,complete,batchID,rewards}.dat	
% 	% cd(parentPath);
% end

% pause(3);

% ------------------------------------------------------------ read sampling ------------------------------------------------------------ %
S = []; Y = []; F = []; C = []; B = [];

cd(parentPath);
% read sequential sampling
for i = 1:numInitPoint
	% system(sprintf('rmdir /s /q %s_Iter%d',modelName,1));
	% system(sprintf('mkdir -p %s_Iter%d',modelName,1));
	currentDirectory = sprintf('%s_Iter%d', modelName, i);
	cd(currentDirectory);
	% local files
	x = dlmread('input.dat'); x = reshape(x, 1, D);
	z = dlmread('projectedInput.dat'); z = reshape(z, 1, d);
	y = dlmread('output.dat');
	f = dlmread('feasible.dat');
	% system('echo 1 > complete.dat'); % deprecated
	c = dlmread('complete.dat');
	%% deprecated: overwrite original files during restart
	% system('echo 0 > batchID.dat'); % deprecated
	batchID = dlmread('batchID.dat');
	% system('echo 0 > acquisitionScheme.dat'); % deprecated
	% global files
	% S = [S; x./xScale]; Y = [Y; y./yScale]; F = [F; f]; C = [C; c]; B = [B; batchID];
	% x = Sinit(i,:); y = Yinit(i,:);
	% f = 1; c = 1; batchID = 0;
	S = [S; z]; Y = [Y; y]; F = [F; f]; C = [C; c]; B = [B; batchID];
	% fprintf('done importing sequential initial iteration %d\n', i);
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
% 		% S = [S; x./xScale]; Y = [Y; y./yScale]; F = [F; f]; C = [C; c]; B = [B; batchID];
% 		S = [S; x]; Y = [Y; y./yScale]; F = [F; f]; C = [C; c]; B = [B; batchID];
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
Z_u = lhsdesign( min(maxInducingPoints, length(S(:,1))), d);
for i = 1:length(Z_u(:,1))
	for j = 1:d
		Z_u(i,j) = zLB(j) + (zUB(j) - zLB(j)) * Z_u(i,j);
	end
end
gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf, 'X_u', Z_u, 'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood');
gp_fic = gp_optim(gp_fic, S, Y, 'opt', opt);

system('rm -fv S.dat Y.dat F.dat C.dat B.dat');
dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('F.dat', F, 'delimiter', '\n');
dlmwrite('C.dat', C, 'delimiter', '\n');
dlmwrite('B.dat', B, 'delimiter', '\n');
toc;
fprintf('Done fitting gp_fic...\n\n\n');

% rescale
% for i = 1:d; S(:,i) = S(:,i) / xScale(i); end; Y = Y / yScale;


% ------------------------------------------------------------ parallel optimization loop ------------------------------------------------------------ %


for i = (numParallelPoint + 1):maxiter


	%% write banner
	fprintf('\n\n---------------------------------------------------------\n');
	fprintf('\nRunning %d concurrent simulations.\n', batchSize);
	fprintf('Exploitation batch size: %d\n', exploitSize);
	fprintf('Exploration batch size: %d\n', exploreSize);
	fprintf('Exploration classification batch size: %d\n', exploreClassifSize);
	fprintf('Iteration: %d\n', i);
	fprintf('\n---------------------------------------------------------\n\n\n');

	cd(parentPath); 
	system('python3 bayesOptSrc/checkComplete.py');
	system('python3 bayesOptSrc/getBatch.py'); % return the batchID for next query; dump to batchID.dat
	batchID = dlmread('batchID.dat');


	%% if there is no free batch then wait
	while batchID == 4
		system('rm -v batchID.dat');
		system('python3 bayesOptSrc/checkComplete.py');
		system('python3 bayesOptSrc/getBatch.py'); % return the batchID for next query; dump to batchID.dat
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
		system('python3 ../bayesOptSrc/getRewards.py'); % return the rewards for sampling acquisition; dump to R.dat
		system('python3 ../bayesOptSrc/getAcquisitionScheme.py'); % dump to acquisitionScheme.dat
		acquisitionFunction = fileread('acquisitionScheme.dat');
		z = getNextSamplingPoint(gp_fic, S, Y, zLB, zUB, acquisitionFunction); % batch acquisition
	elseif batchID == 2
		z = getNextSamplingPointReduceMSE(gp_fic, S, Y, zLB, zUB); % batch explore
	elseif batchID == 3
		% x = getNextSamplingPointReduceMSE(dmodelClassif); % batch exploreClassif
		fprintf('mainprog.m: batchID = 3 is not implemented for high-dimensional problems.\n');
		fprintf('Set exploreClassifSize = 0 to circumvent this error.\n');
	elseif batchID == 4
		fprintf('mainprog.m: getBatch.py: All batches are full\n');
	else
		fprintf('mainprog.m: Error: cannot identify batch\n');
	end

	x = projectEmbedding(A, z, zLB, zUB, xLB, xUB); x = reshape(x, 1, D);
	% dlmwrite('input.dat', x .* xScale, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
	dlmwrite('input.dat', x, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh
	dlmwrite('projectedInput.dat', z, 'delimiter', ',', 'precision', '%0.8e'); % write x to input.dat to be picked up by query.sh

	%% write predictions to gpPredictions.dat
	% [mu, ~, rmse, ~] = predictor(x, dmodel);
	[mu, sigma2] = gp_pred(gp_fic, S, Y, z); sigma2 = sigma2 + gp_fic.lik.sigma2;
	predictFile = fopen('gpPredictions.dat', 'w+');
	% fprintf(predictFile, '%0.8f\n%0.8f\n', mu * yScale, rmse * yScale^2);
	fprintf(predictFile, '%0.8f\n%0.8f\n', mu, sigma2);
	fclose(predictFile);


	%% query functional evaluation
	system('echo 0 > complete.dat'); % echo 0 > complete.dat; indicate case has been queried
	system(sprintf('bash ./%s', queryShellScript)); % end-to-end queryX.sh: from input.dat to {output,feasible,complete,batchID,rewards}.dat
	% note: system() returns MKL errors
	cd(parentPath);
	system('python3 bayesOptSrc/updateDb.py'); % write to {S,Y,F,C}.dat; no B.dat


	%% update inputs and outputs from {input,output,feasible,compelte}.dat at each folder
	S = dlmread('S.dat');
	Y = dlmread('Y.dat');
	F = dlmread('F.dat');
	C = dlmread('C.dat');


	%% generate inducing points
	fprintf('\n\nFitting gp_fic...\n\n\n');

	Z_u = lhsdesign( min(maxInducingPoints, length(S(:,1))), d);
	for i = 1:length(Z_u(:,1))
		for j = 1:d
			Z_u(i,j) = zLB(j) + (zUB(j) - zLB(j)) * Z_u(i,j);
		end
	end

	%% hallucinate and interpolate
	gp_fic_interp = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf, 'X_u', Z_u, 'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood');
	gp_fic_interp = gp_optim(gp_fic_interp, S(F>0,:), Y(F>0), 'opt', opt);
	for i = 1:length(Y)
		if F(i) == 0, [Y(i), ~] = gp_pred(gp_fic_interp, S(F>0,:), Y(F>0), S(i,:)); end
	end
	clear gp_fic_interp;

	%% refit GP
	gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf', gpcf, 'X_u', Z_u, 'jitterSigma2', 1e-4, 'infer_params', 'covariance+likelihood');
	gp_fic = gp_optim(gp_fic, S, Y, 'opt', opt);

	pause(0.25);

end


exit;

