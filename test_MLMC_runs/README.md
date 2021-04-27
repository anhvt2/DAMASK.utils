
# A typical results output

```
[anhtran@solo-login2 test_MLMC_runs]$ python3 wrapper_DREAM3D-DAMASK.py --level=1

wrapper_DREAM3D-DAMASK.py: calling DREAM.3D to generate microstructures
Running test-DownSamplingSVEs-ideapad320-NonExact-base320.json
Output path: /ascldap/users/anhtran/scratch/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs

PipelineRunner Starting. 
   SIMPLib Version 1.2.812.508bf5f37
Loading SIMPLib Plugins.
Adding Folder  "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/bin"
Adding Folder  "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/Plugins"
Adding Folder  "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib"
SIMPL_PLUGIN_PATH: ""
Removed  0  duplicate Plugin Paths
Plugin Directory being Searched:  "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/bin"
Plugin Directory being Searched:  "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/Plugins"
Plugin Directory being Searched:  "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib"
Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libAnisotropy.plugin"
    Pointer:  AnisotropyPlugin(0x10220f0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libDDDAnalysisToolbox.plugin"
    Pointer:  DDDAnalysisToolboxPlugin(0x17f1bc0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libDREAM3DReview.plugin"
    Pointer:  DREAM3DReviewPlugin(0x1845160) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libEMMPM.plugin"
    Pointer:  EMMPMPlugin(0x19a8d20) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libGeneric.plugin"
    Pointer:  GenericPlugin(0x19ec9d0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libHEDMAnalysis.plugin"
    Pointer:  HEDMAnalysisPlugin(0x1a68770) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libImportExport.plugin"
    Pointer:  ImportExportPlugin(0x1b0d420) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libITKImageProcessing.plugin"
    Pointer:  ITKImageProcessingPlugin(0x1ca29a0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libMASSIFUtilities.plugin"
    Pointer:  MASSIFUtilitiesPlugin(0x22891b0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libOrientationAnalysis.plugin"
    Pointer:  OrientationAnalysisPlugin(0x22c14d0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libProcessing.plugin"
    Pointer:  ProcessingPlugin(0x26e6de0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libReconstruction.plugin"
    Pointer:  ReconstructionPlugin(0x27bd120) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libSampling.plugin"
    Pointer:  SamplingPlugin(0x28d2cc0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libSimulationIO.plugin"
    Pointer:  SimulationIOPlugin(0x2996910) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libStatistics.plugin"
    Pointer:  StatisticsPlugin(0x29fb320) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libSurfaceMeshing.plugin"
    Pointer:  SurfaceMeshingPlugin(0x2b8c210) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libSyntheticBuilding.plugin"
    Pointer:  SyntheticBuildingPlugin(0x2cac770) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libTransformationPhase.plugin"
    Pointer:  TransformationPhasePlugin(0x2dedba0) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libUCSBUtilities.plugin"
    Pointer:  UCSBUtilitiesPlugin(0x2e3a450) 

Plugin Being Loaded: "/home/anhtran/data/DREAM.3D/DREAM3D-6.5.138-Linux-x86_64/lib/libZeissImport.plugin"
    Pointer:  ZeissImportPlugin(0x2f71cc0) 

Pipeline Count: 19
: 5%
: [1/19] StatsGenerator 
: 10%
: [2/19] Initialize Synthetic Volume 
: 15%
: [3/19] Establish Shape Types 
: 20%
: [4/19] Pack Primary Phases 
Pack Primary Phases: [4/19] Pack Primary Phases : Packing Features || Initializing Volume
Pack Primary Phases: [4/19] Pack Primary Phases : Packing Features || Placing Features
Pack Primary Phases: [4/19] Pack Primary Phases : Packing Features (1/2) || Generating Feature #100
Pack Primary Phases: [4/19] Pack Primary Phases : Packing Features || Starting Feature Placement...
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #2/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #4/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #6/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #8/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #10/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #12/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #14/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #16/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #18/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #20/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #22/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #24/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #26/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #28/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #30/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #32/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #34/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #36/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #38/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #40/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #42/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #44/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #46/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #48/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #50/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #52/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #54/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #56/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #58/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #60/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #62/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #64/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #66/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #68/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #70/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #72/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #74/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #76/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #78/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #80/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #82/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #84/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #86/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #88/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #90/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #92/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #94/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #96/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #98/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #100/102
Pack Primary Phases: [4/19] Pack Primary Phases : Placing Feature #102/102
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 63/10100 || Est. Time Remain: 00:02:46 || Iterations/Sec: 60.1145
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 164/10100 || Est. Time Remain: 00:02:13 || Iterations/Sec: 74.4102
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 228/10100 || Est. Time Remain: 00:02:20 || Iterations/Sec: 70.3053
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 313/10100 || Est. Time Remain: 00:02:13 || Iterations/Sec: 73.4742
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 411/10100 || Est. Time Remain: 00:02:04 || Iterations/Sec: 77.7378
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 470/10100 || Est. Time Remain: 00:02:08 || Iterations/Sec: 74.7218
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 572/10100 || Est. Time Remain: 00:02:02 || Iterations/Sec: 77.5909
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 628/10100 || Est. Time Remain: 00:02:06 || Iterations/Sec: 74.9582
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 701/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 74.7414
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 748/10100 || Est. Time Remain: 00:02:09 || Iterations/Sec: 72.0547
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 848/10100 || Est. Time Remain: 00:02:04 || Iterations/Sec: 74.3143
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 892/10100 || Est. Time Remain: 00:02:09 || Iterations/Sec: 71.0248
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 942/10100 || Est. Time Remain: 00:02:15 || Iterations/Sec: 67.7893
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1028/10100 || Est. Time Remain: 00:02:12 || Iterations/Sec: 68.5105
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1096/10100 || Est. Time Remain: 00:02:11 || Iterations/Sec: 68.4358
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1141/10100 || Est. Time Remain: 00:02:13 || Iterations/Sec: 67.0348
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1182/10100 || Est. Time Remain: 00:02:16 || Iterations/Sec: 65.5429
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1274/10100 || Est. Time Remain: 00:02:11 || Iterations/Sec: 66.9223
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1320/10100 || Est. Time Remain: 00:02:13 || Iterations/Sec: 65.8551
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1388/10100 || Est. Time Remain: 00:02:12 || Iterations/Sec: 65.9414
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1408/10100 || Est. Time Remain: 00:02:16 || Iterations/Sec: 63.7941
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1444/10100 || Est. Time Remain: 00:02:19 || Iterations/Sec: 61.9822
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1540/10100 || Est. Time Remain: 00:02:15 || Iterations/Sec: 63.3615
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1606/10100 || Est. Time Remain: 00:02:14 || Iterations/Sec: 63.2757
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1655/10100 || Est. Time Remain: 00:02:14 || Iterations/Sec: 62.6751
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1753/10100 || Est. Time Remain: 00:02:10 || Iterations/Sec: 63.9478
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1792/10100 || Est. Time Remain: 00:02:12 || Iterations/Sec: 62.5982
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1844/10100 || Est. Time Remain: 00:02:13 || Iterations/Sec: 61.7053
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 1945/10100 || Est. Time Remain: 00:02:09 || Iterations/Sec: 62.9756
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2014/10100 || Est. Time Remain: 00:02:08 || Iterations/Sec: 63.0814
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2085/10100 || Est. Time Remain: 00:02:06 || Iterations/Sec: 63.3181
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2152/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 63.3892
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2198/10100 || Est. Time Remain: 00:02:06 || Iterations/Sec: 62.3458
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2280/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 62.5583
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2348/10100 || Est. Time Remain: 00:02:03 || Iterations/Sec: 62.7003
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2383/10100 || Est. Time Remain: 00:02:04 || Iterations/Sec: 61.9251
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2448/10100 || Est. Time Remain: 00:02:03 || Iterations/Sec: 61.9825
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2492/10100 || Est. Time Remain: 00:02:03 || Iterations/Sec: 61.4899
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2526/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 60.5901
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2582/10100 || Est. Time Remain: 00:02:04 || Iterations/Sec: 60.157
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2620/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 59.578
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2664/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 59.1645
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2730/10100 || Est. Time Remain: 00:02:04 || Iterations/Sec: 59.1742
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2764/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 58.5854
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2804/10100 || Est. Time Remain: 00:02:05 || Iterations/Sec: 58.0671
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2861/10100 || Est. Time Remain: 00:02:04 || Iterations/Sec: 58.036
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2932/10100 || Est. Time Remain: 00:02:03 || Iterations/Sec: 58.1573
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 2984/10100 || Est. Time Remain: 00:02:02 || Iterations/Sec: 58.0229
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3076/10100 || Est. Time Remain: 00:01:59 || Iterations/Sec: 58.6698
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3132/10100 || Est. Time Remain: 00:01:59 || Iterations/Sec: 58.1702
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3226/10100 || Est. Time Remain: 00:01:56 || Iterations/Sec: 58.7689
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3293/10100 || Est. Time Remain: 00:01:55 || Iterations/Sec: 58.9056
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3332/10100 || Est. Time Remain: 00:01:55 || Iterations/Sec: 58.5023
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3381/10100 || Est. Time Remain: 00:01:55 || Iterations/Sec: 58.3363
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3440/10100 || Est. Time Remain: 00:01:54 || Iterations/Sec: 58.2942
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3520/10100 || Est. Time Remain: 00:01:52 || Iterations/Sec: 58.6227
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3623/10100 || Est. Time Remain: 00:01:49 || Iterations/Sec: 59.3438
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3674/10100 || Est. Time Remain: 00:01:48 || Iterations/Sec: 58.9973
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3786/10100 || Est. Time Remain: 00:01:45 || Iterations/Sec: 59.8133
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3822/10100 || Est. Time Remain: 00:01:45 || Iterations/Sec: 59.3184
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3900/10100 || Est. Time Remain: 00:01:44 || Iterations/Sec: 59.4005
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 3950/10100 || Est. Time Remain: 00:01:44 || Iterations/Sec: 59.1264
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4002/10100 || Est. Time Remain: 00:01:43 || Iterations/Sec: 58.7337
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4064/10100 || Est. Time Remain: 00:01:42 || Iterations/Sec: 58.6131
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4122/10100 || Est. Time Remain: 00:01:42 || Iterations/Sec: 58.5861
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4186/10100 || Est. Time Remain: 00:01:40 || Iterations/Sec: 58.634
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4291/10100 || Est. Time Remain: 00:01:38 || Iterations/Sec: 59.2598
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4351/10100 || Est. Time Remain: 00:01:36 || Iterations/Sec: 59.269
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4394/10100 || Est. Time Remain: 00:01:36 || Iterations/Sec: 58.9656
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4435/10100 || Est. Time Remain: 00:01:36 || Iterations/Sec: 58.7262
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4474/10100 || Est. Time Remain: 00:01:36 || Iterations/Sec: 58.4646
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4550/10100 || Est. Time Remain: 00:01:35 || Iterations/Sec: 58.2624
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4602/10100 || Est. Time Remain: 00:01:34 || Iterations/Sec: 57.9531
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4674/10100 || Est. Time Remain: 00:01:33 || Iterations/Sec: 58.091
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4770/10100 || Est. Time Remain: 00:01:31 || Iterations/Sec: 58.3822
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4848/10100 || Est. Time Remain: 00:01:29 || Iterations/Sec: 58.4463
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 4918/10100 || Est. Time Remain: 00:01:28 || Iterations/Sec: 58.2743
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5038/10100 || Est. Time Remain: 00:01:25 || Iterations/Sec: 58.8716
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5115/10100 || Est. Time Remain: 00:01:24 || Iterations/Sec: 59.0626
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5202/10100 || Est. Time Remain: 00:01:22 || Iterations/Sec: 59.3666
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5274/10100 || Est. Time Remain: 00:01:21 || Iterations/Sec: 59.3885
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5334/10100 || Est. Time Remain: 00:01:20 || Iterations/Sec: 59.2015
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5412/10100 || Est. Time Remain: 00:01:18 || Iterations/Sec: 59.4046
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5463/10100 || Est. Time Remain: 00:01:18 || Iterations/Sec: 59.2947
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5535/10100 || Est. Time Remain: 00:01:16 || Iterations/Sec: 59.3897
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5586/10100 || Est. Time Remain: 00:01:16 || Iterations/Sec: 59.2994
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5664/10100 || Est. Time Remain: 00:01:14 || Iterations/Sec: 59.2952
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5738/10100 || Est. Time Remain: 00:01:13 || Iterations/Sec: 59.4414
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5848/10100 || Est. Time Remain: 00:01:10 || Iterations/Sec: 59.9537
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5935/10100 || Est. Time Remain: 00:01:09 || Iterations/Sec: 60.2092
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 5988/10100 || Est. Time Remain: 00:01:08 || Iterations/Sec: 59.9363
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6044/10100 || Est. Time Remain: 00:01:07 || Iterations/Sec: 59.8861
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6112/10100 || Est. Time Remain: 00:01:06 || Iterations/Sec: 59.948
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6188/10100 || Est. Time Remain: 00:01:05 || Iterations/Sec: 60.0503
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6280/10100 || Est. Time Remain: 00:01:03 || Iterations/Sec: 60.3527
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6382/10100 || Est. Time Remain: 00:01:01 || Iterations/Sec: 60.6602
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6452/10100 || Est. Time Remain: 00:01:00 || Iterations/Sec: 60.7407
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6528/10100 || Est. Time Remain: 00:00:58 || Iterations/Sec: 60.841
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6583/10100 || Est. Time Remain: 00:00:57 || Iterations/Sec: 60.786
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6628/10100 || Est. Time Remain: 00:00:57 || Iterations/Sec: 60.5922
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6718/10100 || Est. Time Remain: 00:00:55 || Iterations/Sec: 60.8013
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6830/10100 || Est. Time Remain: 00:00:53 || Iterations/Sec: 61.2518
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6871/10100 || Est. Time Remain: 00:00:52 || Iterations/Sec: 61.0473
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 6938/10100 || Est. Time Remain: 00:00:51 || Iterations/Sec: 61.0949
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7034/10100 || Est. Time Remain: 00:00:49 || Iterations/Sec: 61.3878
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7098/10100 || Est. Time Remain: 00:00:48 || Iterations/Sec: 61.3065
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7180/10100 || Est. Time Remain: 00:00:47 || Iterations/Sec: 61.2283
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7277/10100 || Est. Time Remain: 00:00:45 || Iterations/Sec: 61.5282
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7372/10100 || Est. Time Remain: 00:00:44 || Iterations/Sec: 61.8062
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7414/10100 || Est. Time Remain: 00:00:43 || Iterations/Sec: 61.4862
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7490/10100 || Est. Time Remain: 00:00:42 || Iterations/Sec: 61.5893
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7534/10100 || Est. Time Remain: 00:00:41 || Iterations/Sec: 61.3818
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7660/10100 || Est. Time Remain: 00:00:39 || Iterations/Sec: 61.8945
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7720/10100 || Est. Time Remain: 00:00:38 || Iterations/Sec: 61.7072
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7782/10100 || Est. Time Remain: 00:00:37 || Iterations/Sec: 61.6753
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7892/10100 || Est. Time Remain: 00:00:35 || Iterations/Sec: 62.0538
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 7972/10100 || Est. Time Remain: 00:00:34 || Iterations/Sec: 62.0766
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8026/10100 || Est. Time Remain: 00:00:33 || Iterations/Sec: 61.9759
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8066/10100 || Est. Time Remain: 00:00:32 || Iterations/Sec: 61.7805
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8124/10100 || Est. Time Remain: 00:00:32 || Iterations/Sec: 61.7475
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8163/10100 || Est. Time Remain: 00:00:31 || Iterations/Sec: 61.5718
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8219/10100 || Est. Time Remain: 00:00:30 || Iterations/Sec: 61.5231
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8330/10100 || Est. Time Remain: 00:00:28 || Iterations/Sec: 61.8549
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8360/10100 || Est. Time Remain: 00:00:28 || Iterations/Sec: 61.5321
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8418/10100 || Est. Time Remain: 00:00:27 || Iterations/Sec: 61.4538
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8490/10100 || Est. Time Remain: 00:00:26 || Iterations/Sec: 61.43
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8546/10100 || Est. Time Remain: 00:00:25 || Iterations/Sec: 61.2568
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8616/10100 || Est. Time Remain: 00:00:24 || Iterations/Sec: 61.3103
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8696/10100 || Est. Time Remain: 00:00:22 || Iterations/Sec: 61.4367
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8820/10100 || Est. Time Remain: 00:00:20 || Iterations/Sec: 61.8466
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8880/10100 || Est. Time Remain: 00:00:19 || Iterations/Sec: 61.8298
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 8984/10100 || Est. Time Remain: 00:00:17 || Iterations/Sec: 62.1197
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9066/10100 || Est. Time Remain: 00:00:16 || Iterations/Sec: 62.2545
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9114/10100 || Est. Time Remain: 00:00:15 || Iterations/Sec: 62.1386
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9238/10100 || Est. Time Remain: 00:00:13 || Iterations/Sec: 62.5508
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9308/10100 || Est. Time Remain: 00:00:12 || Iterations/Sec: 62.5723
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9370/10100 || Est. Time Remain: 00:00:11 || Iterations/Sec: 62.515
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9428/10100 || Est. Time Remain: 00:00:10 || Iterations/Sec: 62.4561
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9513/10100 || Est. Time Remain: 00:00:09 || Iterations/Sec: 62.5781
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9608/10100 || Est. Time Remain: 00:00:07 || Iterations/Sec: 62.7011
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9696/10100 || Est. Time Remain: 00:00:06 || Iterations/Sec: 62.8606
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9826/10100 || Est. Time Remain: 00:00:04 || Iterations/Sec: 63.2434
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 9943/10100 || Est. Time Remain: 00:00:02 || Iterations/Sec: 63.5855
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 10030/10100 || Est. Time Remain: 00:00:01 || Iterations/Sec: 63.7315
Pack Primary Phases: [4/19] Pack Primary Phases : Swapping/Moving/Adding/Removing Features Iteration 10040/10100 || Est. Time Remain: 00:00:00 || Iterations/Sec: 63.3634
Pack Primary Phases: [4/19] Pack Primary Phases : Packing Features || Assigning Voxels
Pack Primary Phases: [4/19] Pack Primary Phases : Packing Features || Assigning Gaps
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 1 || Remaining Unassigned Voxel Count: 3604824
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 2 || Remaining Unassigned Voxel Count: 2954626
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 3 || Remaining Unassigned Voxel Count: 2385117
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 4 || Remaining Unassigned Voxel Count: 1896471
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 5 || Remaining Unassigned Voxel Count: 1487090
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 6 || Remaining Unassigned Voxel Count: 1150338
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 7 || Remaining Unassigned Voxel Count: 876565
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 8 || Remaining Unassigned Voxel Count: 657738
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 9 || Remaining Unassigned Voxel Count: 486117
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 10 || Remaining Unassigned Voxel Count: 353052
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 11 || Remaining Unassigned Voxel Count: 251549
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 12 || Remaining Unassigned Voxel Count: 175657
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 13 || Remaining Unassigned Voxel Count: 119967
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 14 || Remaining Unassigned Voxel Count: 79839
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 15 || Remaining Unassigned Voxel Count: 51790
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 16 || Remaining Unassigned Voxel Count: 32119
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 17 || Remaining Unassigned Voxel Count: 18995
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 18 || Remaining Unassigned Voxel Count: 10608
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 19 || Remaining Unassigned Voxel Count: 5563
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 20 || Remaining Unassigned Voxel Count: 2738
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 21 || Remaining Unassigned Voxel Count: 1300
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 22 || Remaining Unassigned Voxel Count: 616
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 23 || Remaining Unassigned Voxel Count: 289
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 24 || Remaining Unassigned Voxel Count: 111
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 25 || Remaining Unassigned Voxel Count: 27
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 26 || Remaining Unassigned Voxel Count: 2
Pack Primary Phases: [4/19] Pack Primary Phases : Assign Gaps || Cycle#: 27 || Remaining Unassigned Voxel Count: 0
: 25%
: [5/19] Find Feature Neighbors 
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 5.86119% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 11.7197% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 17.5816% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 23.4444% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 29.3057% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 35.1657% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 41.0306% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 46.896% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 52.7624% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 58.6276% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 64.4937% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 70.3596% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 76.2222% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 82.0768% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 87.942% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 93.8074% Complete
Find Feature Neighbors: [5/19] Find Feature Neighbors : Finding Neighbors || Determining Neighbor Lists || 99.6725% Complete
: 30%
: [6/19] Match Crystallography 
Match Crystallography: [6/19] Match Crystallography : Determining Volumes
Match Crystallography: [6/19] Match Crystallography : Determining Boundary Areas
Match Crystallography: Initializing Arrays
Match Crystallography: [6/19] Match Crystallography : Assigning Eulers to Phase 1
Match Crystallography: [6/19] Match Crystallography : Measuring Misorientations of Phase 1
Match Crystallography: [6/19] Match Crystallography : Matching Crystallography of Phase 1
Match Crystallography: [6/19] Match Crystallography : Swapping/Switching Orientations Iteration 14957/100000 || Est. Time Remain: 00:00:05 || Iterations/Sec: 14942.1
: 35%
: [7/19] Generate IPF Colors 
: 40%
: [8/19] Change Resolution 
Change Resolution: [8/19] Change Resolution : Changing Resolution || 0% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 6% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 12% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 18% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 25% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 31% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 37% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 43% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 50% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 56% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 62% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 68% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 75% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 81% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 87% Complete
Change Resolution: [8/19] Change Resolution : Changing Resolution || 93% Complete
Change Resolution: [8/19] Change Resolution : Copying Data...
: 45%
: [9/19] Export DAMASK Files 
: 50%
: [10/19] Change Resolution 
Change Resolution: [10/19] Change Resolution : Changing Resolution || 0% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 12% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 25% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 37% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 50% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 62% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 75% Complete
Change Resolution: [10/19] Change Resolution : Changing Resolution || 87% Complete
Change Resolution: [10/19] Change Resolution : Copying Data...
: 55%
: [11/19] Export DAMASK Files 
: 60%
: [12/19] Change Resolution 
Change Resolution: [12/19] Change Resolution : Changing Resolution || 0% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 3% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 6% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 9% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 12% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 15% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 18% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 21% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 25% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 28% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 31% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 34% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 37% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 40% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 43% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 46% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 50% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 53% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 56% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 59% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 62% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 65% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 68% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 71% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 75% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 78% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 81% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 84% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 87% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 90% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 93% Complete
Change Resolution: [12/19] Change Resolution : Changing Resolution || 96% Complete
Change Resolution: [12/19] Change Resolution : Copying Data...
: 65%
: [13/19] Export DAMASK Files 
: 70%
: [14/19] Change Resolution 
Change Resolution: [14/19] Change Resolution : Changing Resolution || 0% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 1% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 3% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 4% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 6% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 7% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 9% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 10% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 12% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 14% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 15% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 17% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 18% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 20% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 21% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 23% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 25% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 26% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 28% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 29% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 31% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 32% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 34% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 35% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 37% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 39% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 40% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 42% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 43% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 45% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 46% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 48% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 50% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 51% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 53% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 54% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 56% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 57% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 59% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 60% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 62% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 64% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 65% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 67% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 68% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 70% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 71% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 73% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 75% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 76% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 78% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 79% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 81% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 82% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 84% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 85% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 87% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 89% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 90% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 92% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 93% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 95% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 96% Complete
Change Resolution: [14/19] Change Resolution : Changing Resolution || 98% Complete
Change Resolution: [14/19] Change Resolution : Copying Data...
: 75%
: [15/19] Export DAMASK Files 
: 80%
: [16/19] Change Resolution 
Change Resolution: [16/19] Change Resolution : Changing Resolution || 0% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 5% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 10% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 15% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 20% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 25% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 30% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 35% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 40% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 45% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 50% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 55% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 60% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 65% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 70% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 75% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 80% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 85% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 90% Complete
Change Resolution: [16/19] Change Resolution : Changing Resolution || 95% Complete
Change Resolution: [16/19] Change Resolution : Copying Data...
: 85%
: [17/19] Export DAMASK Files 
: 90%
: [18/19] Change Resolution 
Change Resolution: [18/19] Change Resolution : Changing Resolution || 0% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 10% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 20% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 30% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 40% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 50% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 60% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 70% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 80% Complete
Change Resolution: [18/19] Change Resolution : Changing Resolution || 90% Complete
Change Resolution: [18/19] Change Resolution : Copying Data...
: 95%
: [19/19] Export DAMASK Files 
: Pipeline Complete
Microstructure files are generated at:
/ascldap/users/anhtran/scratch/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs


Agent pid 117089
Identity added: /ascldap/users/anhtran/.ssh/id_rsaSolo (/ascldap/users/anhtran/.ssh/id_rsaSolo)
geom_toTable: 
grid     a b c:  8 x 8 x 8
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
vtk_rectilinearGrid: 
geom_toTable: 
grid     a b c:  8 x 8 x 8
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
single_phase_equiaxed_8x8x8.vtk: 729 points and 512 cells...
vtk_addRectilinearGridData: 
adding scalar "microstructure"...
/ascldap/users/anhtran/.local/lib/python2.7/site-packages/vtk/util/numpy_support.py:137: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
  assert not numpy.issubdtype(z.dtype, complex), \
cell mode...
geom_toTable: 
grid     a b c:  10 x 10 x 10
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
vtk_rectilinearGrid: 
geom_toTable: 
grid     a b c:  10 x 10 x 10
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
single_phase_equiaxed_10x10x10.vtk: 1331 points and 1000 cells...
vtk_addRectilinearGridData: 
adding scalar "microstructure"...
/ascldap/users/anhtran/.local/lib/python2.7/site-packages/vtk/util/numpy_support.py:137: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
  assert not numpy.issubdtype(z.dtype, complex), \
cell mode...
geom_toTable: 
grid     a b c:  16 x 16 x 16
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 85
vtk_rectilinearGrid: 
geom_toTable: 
grid     a b c:  16 x 16 x 16
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 85
single_phase_equiaxed_16x16x16.vtk: 4913 points and 4096 cells...
vtk_addRectilinearGridData: 
adding scalar "microstructure"...
/ascldap/users/anhtran/.local/lib/python2.7/site-packages/vtk/util/numpy_support.py:137: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
  assert not numpy.issubdtype(z.dtype, complex), \
cell mode...
geom_toTable: 
grid     a b c:  20 x 20 x 20
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
vtk_rectilinearGrid: 
geom_toTable: 
grid     a b c:  20 x 20 x 20
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
single_phase_equiaxed_20x20x20.vtk: 9261 points and 8000 cells...
vtk_addRectilinearGridData: 
adding scalar "microstructure"...
/ascldap/users/anhtran/.local/lib/python2.7/site-packages/vtk/util/numpy_support.py:137: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
  assert not numpy.issubdtype(z.dtype, complex), \
cell mode...
geom_toTable: 
grid     a b c:  32 x 32 x 32
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
vtk_rectilinearGrid: 
geom_toTable: 
grid     a b c:  32 x 32 x 32
size     x y z:  320.0 x 320.0 x 320.0
origin   x y z:  0.0 : 0.0 : 0.0
homogenization:  1
microstructures: 62
single_phase_equiaxed_32x32x32.vtk: 35937 points and 32768 cells...
vtk_addRectilinearGridData: 
adding scalar "microstructure"...
/ascldap/users/anhtran/.local/lib/python2.7/site-packages/vtk/util/numpy_support.py:137: FutureWarning: Conversion of the second argument of issubdtype from `complex` to `np.complexfloating` is deprecated. In future, it will be treated as `np.complex128 == np.dtype(complex).type`.
  assert not numpy.issubdtype(z.dtype, complex), \
cell mode...
sbatch: INFO: Adding filesystem licenses to job: qscratch:1
In directory: /qscratch/anhtran/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs/10x10x10
done submitting sbatch.damask.solo

INFO:
#SBATCH --nodes=1                     # Number of nodes - all cores per node are allocated to the job
#SBATCH --time=4:00:00               # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
#SBATCH --account=FY210060            # WC ID
#SBATCH --job-name=cpfem              # Name of job
#SBATCH --partition=short,batch       # partition/queue name: short or batch
                                      #            short: 4hrs wallclock limit
                                      #            batch: nodes reserved for > 4hrs (default)
#SBATCH --qos=normal                  # Quality of Service: long, large, priority or normal
                                      #           normal: request up to 48hrs wallclock (default)
                                      #           long:   request up to 96hrs wallclock and no larger than 64nodes
                                      #           large:  greater than 50% of cluster (special request)
                                      #           priority: High priority jobs (special request)
Results available in /qscratch/anhtran/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs/10x10x10

 Elapsed time = 5.01 minutes on Solo
Estimated Yield Stress = 1.9760180000000001 GPa
sbatch: INFO: Adding filesystem licenses to job: qscratch:1
In directory: /qscratch/anhtran/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs/8x8x8
done submitting sbatch.damask.solo

INFO:
#SBATCH --nodes=1                     # Number of nodes - all cores per node are allocated to the job
#SBATCH --time=4:00:00               # Wall clock time (HH:MM:SS) - once the job exceeds this time, the job will be terminated (default is 5 minutes)
#SBATCH --account=FY210060            # WC ID
#SBATCH --job-name=cpfem              # Name of job
#SBATCH --partition=short,batch       # partition/queue name: short or batch
                                      #            short: 4hrs wallclock limit
                                      #            batch: nodes reserved for > 4hrs (default)
#SBATCH --qos=normal                  # Quality of Service: long, large, priority or normal
                                      #           normal: request up to 48hrs wallclock (default)
                                      #           long:   request up to 96hrs wallclock and no larger than 64nodes
                                      #           large:  greater than 50% of cluster (special request)
                                      #           priority: High priority jobs (special request)
Results available in /qscratch/anhtran/DAMASK/DAMASK-2.0.2/examples/SpectralMethod/Polycrystal/testMLMC_14Apr21/DAMASK.utils/test_MLMC_runs/8x8x8

 Elapsed time = 2.34 minutes on Solo
Estimated Yield Stress = 1.9068240000000001 GPa
```
