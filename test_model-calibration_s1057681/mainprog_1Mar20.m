
% ------------------------------------------------------------ construct response GPR(s) ------------------------------------------------------------
close all;
clear all;
home;
format longg;

% log: batch parallel implementation

% variables explanation:
% 	dmodel: hallucinated dmodel -- to be corrected at the end of each iteration
% 	dmodel: real interpolation dmodel (with hallucination in the infeasible regions)
%	dmodelClassif: hallucinated dmodel -- to be corrected at the end of each iteration
%	dmodelClassif: classif GPR

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
%			this is an end-to-end from input.dat to {output,feasible,complete,batchID}.dat for each folder
%			so this queryX.sh must include the post-processing data as well
% Step 4: input file and format
%			(1) create a template at ${modelName}_Template
%			(2) format all the iteration as ${modelName}_Iter{1,2,3,...}
% Step 5: post-process: use buildLogs.sh to build globale {input,output,feasible,complete,batchID}.dat for post-processing analysis
%

% ------------------------------------------------------------ input parameters ------------------------------------------------------------
% simulation settings
parentPath = pwd; % no / at the end
modelName = 'spkWeld'; % declare model name -- must match with "${modelName}_Template/"
queryShellScript = 'queryFPGA.sh' % query Shell script -- end-to-end, from input.dat to {output,feasible,complete,batchID,rewards}.dat

cd(parentPath); % change to parent path
addpath(parentPath); % add current path
addpath(strcat(parentPath,'/dace')); % add GPR toolbox
addpath(strcat(parentPath,'/bayesSrc')); % add BO toolbox

% define lower and upper bounds for the control variables -- tuned on 21Dec17 based on 481 random initial sampling cases
xLB = [15, 120, 100, 250]; % xLB = [15, 120, 50];
xUB = [30, 200, 300, 650]; % xUB = [30, 200, 250];

d = length(xLB); % dimensionality of the problem

% define objective GPR params
% initial guess for theta hyper-parameters
theta = 1 * ones(1,d);
lob = 2.5e-1 * ones(1,d); 
upb = 7.5e+0 * ones(1,d); 

% feasibility classif-GPR
thetaClassif = 1e+0 * ones(1, d); % classif-GPR hyper-params
lobClassif = 1e-2 * ones(1, d); % lowerbounds classif-GPR hyper-params
upbClassif = 2e+1 * ones(1, d); % upperbounds classif-GPR hyper-params

% batch setting
exploitSize = 20; % exploitation by hallucination in batch
exploreSize = 5; % exploration by sampling at maximal mse
exploreClassifSize = 0; % exploration by sampling at maximal for classif-GPR
batchSize = exploitSize + exploreSize + exploreClassifSize; % total number of concurrent simulations

% optimization settings
maxiter = 8000; % maximum number of iterations
numInitPoint = 999; % 7; % last maximum number of iterations in the initial sampling phase
numParallelPoint = numInitPoint; % true for asynchornously batch-parallel  % last maximum number of iterations in the batch parallel BO; constraint numParallelPoint >= numInitPoint; (cont)
%% if no parallel for batch parallel BO, then numParallelPoint = numInitPoint
fprintf('Running Bayesian-optimization on %s\n', modelName);
checkTime = 2; % minutes to periodically check simulations if they are complete in MATLAB master optimization loop
waitTime = 1; % hours to stop waiting for a batch to finish; indeed run post-processing after this
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

% ------------------------------------------------------------ read sampling ------------------------------------------------------------
S = []; Y = []; F = []; C = []; B = [];

cd(parentPath);
% read sequential sampling
for i = 1:numInitPoint
	% system(sprintf('rmdir /s /q %s_Iter%d',modelName,1));
	% system(sprintf('mkdir %s_Iter%d',modelName,1));
	currentDirectory = sprintf('%s_Iter%d', modelName, i);
	cd(currentDirectory);
	% local files
	x = dlmread('input.dat'); x = reshape(x, 1, length(x));
	y = dlmread('output.dat');
	feasible = dlmread('feasible.dat');
	% system('echo 1 > complete.dat'); % deprecated
	c = dlmread('complete.dat');
	%% deprecated: overwrite original files during restart
	% system('echo 0 > batchID.dat'); % deprecated
	batchID = dlmread('batchID.dat');
	% system('echo 0 > acquisitionScheme.dat'); % deprecated
	% global files
	S = [S; x]; Y = [Y; y]; F = [F; feasible]; C = [C; c]; B = [B; batchID];
	fprintf('done importing sequential initial iteration %d\n', i);
	cd(parentPath);
end

% read parallel sampling -- WARM RESTART
for i = (numInitPoint + 1):numParallelPoint
	folderList = dir(sprintf('%s_Iter%d*', modelName, i));
	for j = 1:length(folderList)
		currentDirectory = folderList(j).name;
		cd(currentDirectory);
		% local files
		x = dlmread('input.dat'); x = reshape(x, 1, length(x));
		y = dlmread('output.dat'); 
		f = dlmread('feasible.dat');
		system('echo 1 > complete.dat'); c = dlmread('complete.dat');
		batchID = dlmread('batchID.dat');
		% global files
		S = [S; x]; Y = [Y; y]; F = [F; f]; C = [C; c]; B = [B; batchID];
		fprintf('done importing parallel BO iteration %d folder %s\n', i, currentDirectory);
		cd(parentPath);
	end
end

% before interpolate
dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('F.dat', F, 'delimiter', ',', 'precision', '%d');
dlmwrite('C.dat', C, 'delimiter', ',', 'precision', '%d');
dlmwrite('B.dat', B, 'delimiter', ',', 'precision', '%d');
l = length(Y);

% interpolate using feasible and hallucinate at infeasible regions
tic;
fprintf('\n\nFitting dmodel...\n\n\n');
[dmodelInterp, ~] = dacefit(S(F>0,:), Y(F>0), @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
for i = 1:length(Y)
	if F(i) == 0, Y(i) = predictor(S(i,:), dmodelInterp); end
end
clear dmodelInterp;
[dmodel, ~] = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
[dmodelClassif, ~] = dacefit(S, F, @regpoly0, @corrgauss, thetaClassif, lobClassif, upbClassif, xLB, xUB);

% write to files
system('rm -fv S.dat Y.dat F.dat C.dat B.dat');
dlmwrite('S.dat', S, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('Y.dat', Y, 'delimiter', ',', 'precision', '%0.16f');
dlmwrite('F.dat', F, 'delimiter', '\t');
dlmwrite('C.dat', C, 'delimiter', '\t');
dlmwrite('B.dat', B, 'delimiter', '\t');
toc;
fprintf('Done fitting dmodel...\n\n\n');


% ------------------------------------------------------------ parallel optimization loop ------------------------------------------------------------


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
	system('python3 bayesSrc/checkComplete.py');
	system('python3 bayesSrc/getBatch.py'); % return the batchID for next query; dump to batchID.dat
	batchID = dlmread('batchID.dat');


	%% if there is no free batch then wait
	while batchID == 4
		system('rm -v batchID.dat');
		system('python3 bayesSrc/checkComplete.py');
		system('python3 bayesSrc/getBatch.py'); % return the batchID for next query; dump to batchID.dat
		batchID = dlmread('batchID.dat');
		pause(checkTime * 60);
	end

	currentFolder = sprintf('%s_Iter%d', modelName, i);
	system(sprintf('mkdir %s', currentFolder)); % system([fprintf('cp -rfv %s_Template %s', modelName, currentFolder)]);
	cd(currentFolder);
	system(sprintf('cp ../%s_Template/* .', modelName));
	system('mv -v ../batchID.dat .'); % output of getBatch.py
	

	%% sample acquisition scheme based on rewards
	if batchID == 1
		system('python3 ../bayesSrc/getRewards.py'); % return the rewards for sampling acquisition; dump to R.dat
		system('python3 ../bayesSrc/getAcquisitionScheme.py'); % dump to acquisitionScheme.dat
		acquisitionFunction = fileread('acquisitionScheme.dat');
		x = getNextSamplingPoint(dmodel, dmodelClassif, acquisitionFunction); % batch acquisition
	elseif batchID == 2
		x = getNextSamplingPointReduceMSE(dmodel); % batch explore
	elseif batchID == 3
		x = getNextSamplingPointReduceMSE(dmodelClassif); % batch exploreClassif
	elseif batchID == 4
		fprintf('mainprog.m: getBatch.py: All batches are full\n');
	else
		fprintf('mainprog.m: Error: cannot identify batch\n');
	end
	dlmwrite('input.dat', x, 'delimiter', ',', 'precision', '%0.8f'); % write x to input.dat to be picked up by query.sh
	
	
	%% write predictions to gpPredictions.dat
	[mu, ~, rmse, ~] = predictor(x,dmodel);
	predictFile = fopen('gpPredictions.dat', 'w+');
	fprintf(predictFile, '%0.8f\n%0.8f\n', mu, rmse);
	fclose(predictFile);
	

	%% query
	system('echo 0 > complete.dat'); % echo 0 > complete.dat; indicate case has been queried
	system(sprintf('bash ../', queryShellScript)); % end-to-end queryX.sh: from input.dat to output.dat, feasible.dat, complete.dat, rewards.dat
	% note: system() returns MKL errors
	cd(parentPath);
	system('python3 updateDb.py'); % write to {S,Y,F,C}.dat; no B.dat
	
	%% update
	S = dlmread('S.dat');
	Y = dlmread('Y.dat');
	F = dlmread('F.dat');
	C = dlmread('C.dat');


	%% fit dmodel/hallucinate
	fprintf('\n\nFitting dmodel...\n\n\n');
	[dmodelInterp, ~] = dacefit(S(F>0,:), Y(F>0), @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
	for i = 1:length(Y)
		if F(i) == 0, Y(i) = predictor(S(i,:), dmodelInterp); end
	end
	clear dmodelInterp;
	[dmodel, ~] = dacefit(S, Y, @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
	[dmodelClassif, ~] = dacefit(S, F, @regpoly0, @corrgauss, thetaClassif, lobClassif, upbClassif, xLB, xUB);


end


