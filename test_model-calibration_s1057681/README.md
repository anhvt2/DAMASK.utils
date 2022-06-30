# aphBO-2GP-3B: An synchronous parallel constrained multi-acquisition function Bayesian optimization framework on high-performance platforms

**Author: Anh Tran (SNL) -- <anhtran@sandia.gov>, <anh.vt2@gmail.com>**

## Prerequisite

1. Python
	* numpy
	* natsort
	* matplotlib
	* datetime
	

1. Matlab: should be self-contained, no extra toolbox is required
	* ~~may require Parallel Computing Toolbox for Big Data~~
	* can be replaced with Octave using `gpml` toolbox

1. Octave
	* download source
	* `./configure --prefix=/usr/local/` if sudo
	* `./configure --prefix=$HOME` if not sudo
	* in terminal: `sudo apt install liboctave-dev` if sudo
	* in Octave
		* `pkg install -forge io`
		* `pkg install -forge statistics`

## Release history

By default, the package solves for the <b>maximization</b> problem. If minimization is desired, then (a) first multiply with (-1) before solving, and (b) then multiply with (-1) in the post-process.

* 0.0: ```daceParallelTemplate_*```
to-do list for ```daceParallelTemplate_*```
	1.	asynchronous parallel feature
	2.	bug: fix UCB acquisition function 
	3.	IMSE: cannot implement so far for the same theta (```*_30Sep18```)
	4.	scalable/local: domain decomposition approach

* 0.1: ```daceAsyncParTemplate_13Oct18```:
	1. 	decompose calcNegAcquis.m to different flavors for GP-Hedge
	2. 	aim to implement asynchronous

* 0.2: ```ooDaceMfSeqConstrainedTemplate_1Sep19```:
	1. 	```bayesSrc/calculateNegativeAcquisitionFunction.m```:
		* line 4-8: add safeguards:
		```matlab
		% probability of feasibility
		x = reshape(x, 1, length(x)); % reshape to row vector
		[feasibleProb, s2FeasibleProb] = ooDmodelClassif.predict(x); 
		if feasibleProb > 1; feasibleProb = 1; end
		if feasibleProb < 0; feasibleProb = 0; end
		```
	2.  ```bayesSrc/calculateNegativeAcquisitionFunction.m```:
		* line 44-45: more elaboration:
		```matlab
		fprintf('Acquisition %s: mu = %0.8f\n', acquisitionFunction, y);
		fprintf('Acquisition %s: sigma = %0.8f\n', acquisitionFunction, s2);
		if strcmp(acquisitionFunction, 'UCB')
			fprintf('Acquisition %s: kappa = %0.8f\n', acquisitionFunction, kappa);
		end
		```
	3. line 38: make sure this is '+', not '-'
		```matlab
		a = y + sqrt(kappa) * sqrt(s2);
		```

* 0.3: ```ooDaceMfSeqConstrainedTemplate_19Feb19```:
	1.	```@BasicGaussianProcess/imse.m```:
		* line 74-83: add Monte Carlo sampling on high-dimensional
		```matlab
		%> @todo Implement generic monte carlo integration
		% error( 'BasicGaussianProcess:imse: Not supported for input dimension > 3.' );
		% edit by AT
		nMonteCarlo = 5e3; tmpSum = 0; 
		for i = 1:nMonteCarlo
			x = lb + (ub - lb) .* rand(size(ub));
			[~, s2] = this.predict(x);
			tmpSum = tmpSum + s2;
		end
		out = (tmpSum / nMonteCarlo) * prod(ub - lb);
		```
	2. 	```bayesSrc/calculateNegativeAcquisitionFunction.m```
		* line 33: change ooDmodel to optsOoDace for assessing xUB and xLB:
		```matlab
		volD = prod(optsOoDace.xUB - optsOoDace.xLB);
		```

	3. 	```bayesSrc/calculateNegativeAcquisitionFunction.m```
		* line 37-38: add kappa = constant for UCB acquisition function
		```matlab
		%% option-3: kappa = constant
		kappa = 2;
		```

* 0.4: ```daceAsynchParHedgeTemplate_9Sep19```:
	1. line 29-30: ```bayesSrc/getNextSamplingPoint.m```: tune sigma (as default opt) in CMAES
		```matlab
		d = length(xUB);
		sigma = 1/3  * reshape(xUB - xLB, d, 1);
		```
	2. line 31-32: ```bayesSrc/calculateNegativeAcquisitionFunction.m```: add kappa = constant option
		```matlab
		%% option-3: kappa = constant
		kappa = 2;
		```
	3. line 40-44: ```bayesSrc/calculateNegativeAcquisitionFunction.m```: print out more details
		```matlab
		fprintf('Acquisition %s: mu = %0.8f\n', acquisitionFunction, y);
		fprintf('Acquisition %s: sigma = %0.8f\n', acquisitionFunction, rmse);
		if strcmp(acquisitionFunction, 'UCB')
			fprintf('Acquisition %s: kappa = %0.8f\n', acquisitionFunction, kappa);
		end
		```
	4. line 5-10: ```bayesSrc/calculateNegativeAcquisitionFunction.m```: add safeguard for feasProb
		```matlab
		% safeguard
		if feasProb > 1
			feasProb = 1;
		elseif feasProb < 0
			feasProb = 0;
		end
		```
	5. ```mainprog_25Jun19.m```: FPGA example
	6. ```mainprog_26Jun19.m```: SPPARKS example; may not be compatible

* 0.5: update on GitHub repository (1Mar20)
	1. ```bayesSrc/calculateNegativeAcquisitionFunction.m```: add safeguard for rmse < 0 (line 19)
		```matlab
		if rmse < 0; rmse = 0; end
		```
	2. fix a crucial bug: outputs are not looped in the ```mainprog.m```:
		```matlab
		% query
		system('echo 0 > complete.dat'); % echo 0 > complete.dat; indicate case has been queried
		system('bash ../querySpk.sh'); % -- end-to-end, from input.dat to {output,feasible,complete,batchID,rewards}.dat
		% note: system() returns MKL errors
		cd(parentPath);
		% update
		system('python updateDb.py'); % write to {S,Y,F,C}.dat; no B.dat
		S = dlmread('S.dat'); % << corrected
		Y = dlmread('Y.dat'); % << corrected
		F = dlmread('F.dat'); % << corrected
		C = dlmread('C.dat'); % << corrected
		% fit dmodel
		fprintf('\n\nFitting dmodel...\n\n\n');
		[dmodelInterp, ~] = dacefit(S(F>0,:), Y(F>0), @regpoly0, @corrgauss, theta, lob, upb, xLB, xUB);
		```
	3. fix feasible (f=0) if ```feasible.dat``` not found 
		```python
		if os.path.isfile(parentPath + '/' + folderList[i] + 'feasible.dat'):
			f = np.loadtxt(parentPath + '/' + folderList[i] + 'feasible.dat')
		else: 
			f = 0
		```
	4. ```updateDb.py``` does not work for d=1 parameter. ```len()``` returns error. 
	5. change ```python``` to ```python3```

* 0.6: update on GitHub repository (12Jun20)
	1. add ```xScale``` and ```yScale``` in ```mainprog.m``` as a part of input parameters: to avoid situation when input and output varies from 1e-6 to 1e+6 order of magnitude, but preserve ```input.dat``` and ```output.dat``` as from simulations. Rescaling is done internally.
	2. add rescale for ```xLB``` and ```xUB``` lines 128-129:
		```matlab
		xLB = xLB ./ xScale; % rescale
		xUB = xUB ./ xScale; % rescale
		```

* 0.7: change order in `utils/getBatch.py` to promote exploration over exploitation.

* 0.8: bug fix and update on GitHub repository (6Nov21)
	1. include `dace`, `ooDACE`, `gpml` (octave), and `GPstuff` (octave) in the future
	2. fix several bugs related to the acquisition function
	3. successful test
	4. default to `aphBO` framework


## Explanation

1. ```acquisitionScheme.dat```: self-explanatory, can be either ```PI```, ```EI```, or ```UCB```.

2. ```R.dat```: rewards tracking file for each iteration, for sampling acquisition function
	* line 1: rewards for the UCB acquisition function
	* line 2: rewards for the EI acquisition function
	* line 3: rewards for the PI acquisition function


## BUG list

1. add a `- min(Y)` before fitting the GP, because right now the observation is not strictly positive. If it is not, in the constraint handling scheme, the maximum of acquisition could go to constrained region, and as such, the algorithm fails to optimize. This is a serious bug.

1. implement `bayesOptSrc-gpml/` and `mainprog.m` for [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/) toolbox. One of the main reasons for switching to [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/) is that it supports Octave (open-source) as opposed to commercial MATLAB license. 

1. ~~print out post-processing analysis after the optimization run.~~
	* ~~visual step plot for convergence study~~ (```bayesSrc/visualResTestPlot.py```)
	* ~~print out the associated folder with convergent iterations~~(```bayesSrc/visualResTestPlot.py```)
	* ~~document run time, e.g. because of what iteration finished, then another iteration has started. This will create a runtime analysis for evaluating the numerical performance.~~ (see comment #5 in the same section)

1. ~~implement ```StandardScaler``` to scale input that runs at different scales: from 1e-6 to 1e6 is too wide of a range.~~ (see ```xScale``` and ```yScale``` in ```mainprog.m```)

1. add forensic analysis to ```bayesSrc/updateDb.py``` for simulations that are cut off due to passing time limitation. 

1. ~~move ```checkConstraint.m``` to ```X_Template``` (and remove ```bayesSrc/checkConstraint.m```)~~ (application-dependent)

1. ~~benchmark by modification and created time~~ (now punch time card in ```query.log```)
	```python
	import os.path, time
	print("last modified: %s" % time.ctime(os.path.getmtime(file)))
	print("created: %s" % time.ctime(os.path.getctime(file)))
	```

1. ~~create a performance assessment in ```utils/``` by looking at the performance analysis (run time vs. idle time between pBO and aphBO)~~ see `utils/compareConvergence_ByTime_bench.py` and `utils/compareConvergence_ByIter_bench.py`

1. ~~implement a filter for ```S``` to avoid multiple design sites error for duplicating inputs~~ (```checkMultipleDesignSites.m``` and ```getNextSamplingPoint.m```)

1. ~~debug the rewarding scheme (```getRewards.py``` and ```getAcquisitionScheme.py```) with stabilization in ```getAcquisitionScheme.py```, i.e. ```np.exp(1000) / np.exp(1005) = np.exp(-5))```, in practice it is ```np.nan```)~~

1. ~~add ```imp3d_Template/``` for impeller design optimization~~

1. add ```bayesSrc/getNextSamplingPointReduceIMSE.m``` for integrated mean-square error (IMSE), could be further extended as another acquisition function to compete in the acquisition function sampling scheme

1. ~~fix ```utils/plotConvergence.py``` for maximization settings~~


## How to set up a case study

By default, the package solves for the <b>maximization</b> problem. If minimization is desired, then (a) first multiply with (-1) before solving, and (b) then multiply with (-1) in the post-process.

1. Run the *initial sampling* (MC/LHS). Make sure the following files are in each of initial sampling folders. (use ```utils/prepareInitialSetup.sh``` after set ```numInitPoint``` parameter to the number of initial sampling folders)

```shell
Output:
echo 0 > batchID.dat
echo 0 > acquisitionScheme.dat
```

~~*This can be done in ```mainprog.m```*~~ This has to be done manually (by changing the ```numInitPoint``` parameter in ```mainprog.m```) because after a warm restart, the optimizer will overwrite valuable information, such as batch and acquisition scheme for previous runs. This effect is undesirable. 

2. Change the following parameters in ```mainprog.m```

```matlab
...

modelName = 'fpga'; % declare model name -- must match with "${modelName}_Template/"
queryShellScript = 'queryFPGA.sh'; % query Shell script -- end-to-end, from input.dat to {output,feasible,complete,batchID,rewards}.dat

%% define lower and upper bounds for the control variables
xLB	= [20000, 300, 30000, 100 , 10e-6, 2000, 100 , 8.0e-6 , 1.0, 0.5, 12.0e-6];
xUB	= [30000, 750, 40000, 1800, 17e-6, 6000, 2500, 25.0e-6, 3.0, 1.0, 16.7e-6];

% add rough scale of inputs and outputs so that it can be centered around 1e0
xScale = [1e4  , 1e2, 1e4,   1e3 , 1e-6 , 1e3 , 1e3 , 1e-6   , 1e0, 1e0, 1e-6   ];
yScale = [1e-4];

...

%% batch-size setting
exploitSize = 20;	   % exploitation by hallucination in batch
exploreSize = 5;		% exploration by sampling at maximal mse
exploreClassifSize = 0; % exploration by sampling at maximal for classif-GPR
batchSize = exploitSize + exploreSize + exploreClassifSize; % total number of concurrent simulations

%% optimization settings
maxiter = 8000; % maximum number of iterations
numInitPoint = 3; % last maximum number of iterations in the initial sampling phase


```

3. Implement an end-to-end ```queryX.sh``` shell script that takes from ```input.dat``` to ```{output,feasible,complete,batchID,rewards}.dat```; also dumps out ```query.log``` for forensic analysis **NOTE: for querying on HPC, print ```query.log``` from the ```sbatch.*``` or ```qsub.*``` script instead**: could be done by [utils/writeQueryLog.py](https://github.com/anhvt2/daceAsyncParHedge.GitHub/blob/master/utils/writeQueryLog.py) as simple as 
```python
python3 ../utils/writeQueryLog.py; 
# run simulations
python3 ../utils/writeQueryLog.py; 
```

(**note: `writeQueryLog.py` logs the beginning and the end of the process for tracking purposes.**)

or

```shell
...
# date +%Y-%m-%d-%H:%M:%S
timeStamp=$(date +%Y-%m-%d-%H:%M:%S)
logFile="query.log"
echo "Start querying at" > query.log
echo ${timeStamp} >> query.log
...
timeStamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "Stop querying at" >> query.log
echo ${timeStamp} >> query.log
```

## Benchmark

A lot of of benchmarking functions are adopted from [https://www.sfu.ca/~ssurjano/optimization.html](https://www.sfu.ca/~ssurjano/optimization.html). For full implementation of benchmarking BO methods, see [this GitHub repository](https://github.com/anhvt2/benchBO) instead.

1. Use ```mainprog_bench.m``` for benchmark

2. For R scripts:
	* Convention: 
		* ```*.R``` is the script that should be used for benchmark
		* ```*.r``` is the template script that should **not** be used for benchmark
	* in ```bench.R```, create  a ```.r``` benchmarking function
	* in ```R_Template```, do 
		```shell
		cd R_Template/
		ln -sf ../bench.R/*.r .
		python3 writeRTestFiles.py
		cd ..
		```
		* ```writeRTestFiles.py``` will create ```test_X.R```, which reads ```input.dat``` (comma as delimiter) and dumps ```{output,feasible,complete}.dat```
		* ```test_X.R``` is supposed to be called within ```queryX.sh```

3. ~~To select asynchronous parallel implementation with only one acquisition function, modify ```bayesSrc/getAcquisitionScheme.py```~~```bayesSrc/getAcquisitionScheme{UCB,EI,PI}.py```

4. Select ```cmaes.m``` hyper-parameters (increase ```MaxIter``` and ```MaxFunEvals``` in ```getNextSamplingPoint.m``` for high-fidelity applications)
	```matlab
	OPTS.MaxIter = 5000; % set MaxIter
	OPTS.MaxFunEvals = 5000; % set MaxFunEvals
	```

5. For comparison of numerical performance, we provide a benchmark of 10 benchmark functions (all written in R), including
	* random search, i.e. Monte Carlo
	* cBO (classical BO) (could be achieved by either aphBO-2GP-3B or pBO-2GP-3B with batch-size reduced to 1)
		* UCB
		* EI
		* PI
	* pBO-2GP-3B -- batch-sequential parallel
		* UCB
		* EI
		* PI
	* apBO-2GP-3B -- no GP-Hedge, asynchronous parallel
		* UCB
		* EI
		* PI
	* aphBO-2GP-3B -- with GP-Hedge, asynchronous parallel

	which total up to comparison of 11 methods for the same benchmark function. Each function may repeat several times, 5 or 10 times.

6. Organization of the benchmark, from high-level to low-level:
	* high-level: name of folder after benchmark function, e.g. ```camel3```
		* function to benchmark, i.e. ```camel3.r```
	* low-level: name of methods, i.e. ```pBO-2GP-3B_UCB```
		* should be highly automated~~, i.e. change of ```${modelName}``` in a few places should be sufficient~~
		* automatic initial sampling on-the-fly, preferably the same across different methods (may have to initialize first for every benchmark functions, e.g. 2 initial samples?)
		* swift change of test function within one low-level folder (think about a Shell script that does everything given the ```${modelName}```)

7. Implementing a large-scale benchmark:
	1. first of all, have a template ready for each method
	2. change method to different benchmark functions:
		1. change of ```modelName``` in ```mainprog.m```
		2. change ```queryShellScript``` in ```mainprog.m```
		3. change support domain
			```matlab
			...
			xLB	= [20000, 300, 30000, 100 , 10e-6, 2000, 100 , 8.0e-6 , 1.0, 0.5, 12.0e-6];
			xUB	= [30000, 750, 40000, 1800, 17e-6, 6000, 2500, 25.0e-6, 3.0, 1.0, 16.7e-6];
			...
			xScale = [1e4  , 1e2, 1e4,   1e3 , 1e-6 , 1e3 , 1e3 , 1e-6   , 1e0, 1e0, 1e-6   ];
			yScale = [1e-4];
			...
			```
		4. if ```queryR.sh``` is used, then change the script name, i.e. ```test_camel3.R```
			```shell
			...
			gnome-terminal -- timeout ${waitTime}h bash -c "R < test_camel3.R --no-save; echo 1 > complete.dat" > log.timeout
			...
			```
		5. fix the ```maxiter``` (around 100-500) and make sure that all methods have the same initial sampling (for each run)
			```matlab
			maxiter = 8000; % maximum number of iterations
			numInitPoint = 3
			```

## References

1. Practical implementation detail of Bayesian Optimization: [https://stats.stackexchange.com/questions/194166](https://stats.stackexchange.com/questions/194166)

1. Another choice for ```kappa``` in the GP-UCB acquisition function is described in [https://arxiv.org/pdf/2006.10948.pdf](https://arxiv.org/pdf/2006.10948.pdf) and in [the original GP-UCB paper](https://www.researchgate.net/publication/241638164)
	```matlab
	kappa = 2*log(t^2 * 2 * pi^2 / 3 / delta ) + 2*d*log(t^2 * d * b * r * sqrt(log( 4*d*a/delta)))
	% a,b: some constants; see bounds on Theorem 2 in Srinivas et al. GP-UCB
	% delta \in (0,1)
	```

## Citations

1. Methodology: preprint on arXiv: [https://arxiv.org/pdf/2003.09436.pdf](https://arxiv.org/pdf/2003.09436.pdf)

> Tran, A., McCann, S., Furlan, J. M., Pagalthivarthi, K. V., Visintainer, R. J., & Wildey, T. (2020). aphBO-2GP-3B: A budgeted asynchronously-parallel multi-acquisition for constrained Bayesian optimization on high-performing computing architecture. arXiv preprint arXiv:2003.09436. (accepted to SAMO)


1. ICME Application to [Sandia/SPPARKS kMC](https://www.sciencedirect.com/science/article/pii/S1359645420303220)

> Tran, A., Mitchell, J. A., Swiler, L., & Wildey, T. (2020). An active learning high-throughput microstructure calibration framework for solving inverse structure-process problems in materials informatics. Acta Materialia 2020 (194) 80-92.



