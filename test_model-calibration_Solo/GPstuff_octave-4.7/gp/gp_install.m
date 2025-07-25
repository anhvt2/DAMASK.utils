function gp_install(suiteSparse)
%  Matlab function to compile all the c-files to mex in the GPstuff/gp
%  folder.  The function is called from GPstuff/gpstuff_install.m but
%  can be run separately also.
%
%  If you want to use GPstuff without compactly supported (CS)
%  covariance functions run as gp_install([]). If you want to use CS
%  covariance functions read further.
%
%  Some of the sparse GP functionalities in the toolbox require
%  SuiteSparse toolbox by Tim Davis. First install SuiteSparse from:
%    http://www.cise.ufl.edu/research/sparse/SuiteSparse/current/SuiteSparse/
%
%  Note! Install also Metis 4.0.1 as mentioned under header "Other
%        packages required:".           
%
%  After this, compile the c-files in GPstuff/gp as follows:
%
%   Run gp_install( suitesparse_path ) in the present directory. 
%   Here suitesparse_path is a string telling the path to SuiteSparse 
%   package (for example, '/matlab/toolbox/SuiteSparse/'). Note! It is
%   important that suitesparse_path is in right format. Include also
%   the last '/' sign in it.
    
% Parts of the installation code are modified from the 
% CHOLMOD/MATLAB/cholmod_install.m file in the SuiteSparse version 3.2.0.

%   Copyright (c) 2006-2007, Timothy A. Davis
%   Copyright (c) 2008-2010 Jarno Vanhatalo
    
% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

% Compile the 'dist_euclidean' mex-function
if (~isempty (strfind (computer, '64')))
  % 64-bit MATLAB
  if ~exist('OCTAVE_VERSION','builtin')
    mex -O -g -largeArrayDims -output private/dist_euclidean linuxCsource/dist_euclidean.c
  else
    mex --output private/dist_euclidean.mex linuxCsource/dist_euclidean.c
  end
else
  if ~exist('OCTAVE_VERSION','builtin')
    mex -O -output private/dist_euclidean linuxCsource/dist_euclidean.c
  else
    mex --output private/dist_euclidean.mex linuxCsource/dist_euclidean.c
  end
end

if nargin<1 || isempty(suiteSparse)
    % Compile without SuiteSparse.
    % This means that compactly supported covariance functions can not be used.
    % These are: gpcf_ppcs0, gpcf_ppcs1, gpcf_ppcs2, gpcf_ppcs3
    
    % Compile the 'trcov' mex-function
    if (~isempty (strfind (computer, '64')))
      % 64-bit MATLAB
      if ~exist('OCTAVE_VERSION', 'builtin')
        mex -O -g -largeArrayDims -output private/trcov linuxCsource/trcov.c
        mex -O -g -largeArrayDims -output private/dist_euclidean linuxCsource/dist_euclidean.c
      else
        mex --output private/trcov.mex linuxCsource/trcov.c
        mex --output private/dist_euclidean.mex linuxCsource/dist_euclidean.c
      end
    else
      if ~exist('OCTAVE_VERSION', 'builtin')
        mex -O -output private/trcov linuxCsource/trcov.c
        mex -O -output private/dist_euclidean linuxCsource/dist_euclidean.c
      else
        mex --output private/trcov.mex linuxCsource/trcov.c
        mex --output private/dist_euclidean.mex linuxCsource/dist_euclidean.c
      end
    end
    
    fprintf ('\n GP package succesfully compiled ') ;
    fprintf ('\n   without compactly supported covariance functions\n') ;
    
    
else
    v = getversion ;

    details = 0 ;	    % 1 if details of each command are to be printed

    try
        % ispc does not appear in MATLAB 5.3
        pc = ispc ;
    catch
        % if ispc fails, assume we are on a Windows PC if it's not unix
        pc = ~isunix ;
    end

    d = '' ;
    if (~isempty (strfind (computer, '64')))
        % 64-bit MATLAB
        d = '-g -largeArrayDims' ;
        
        % Compile the 'trcov' mex-function
        mex -O -g -largeArrayDims -output private/trcov linuxCsource/trcov.c 
        mex -O -g -largeArrayDims -output private/ldlrowmodify linuxCsource/ldlrowmodify.c 
        
        if v >= 7.8
            d = [d ' -DLONG -D''LONGBLAS=UF_long'''];
        end
    else
        mex -O -output private/trcov linuxCsource/trcov.c 
        mex -O -output private/ldlrowmodify linuxCsource/ldlrowmodify.c 
    end

    % Compile the 'spinv' and 'ldlrowupdate' mex-functions
    % This is awfully long since the functions need all the functionalities of SuiteSparse

    include = '-I../../CHOLMOD/MATLAB -I../../AMD/Include -I../../COLAMD/Include -I../../CCOLAMD/Include -I../../CAMD/Include -I../Include -I../../UFconfig' ;

    if (v < 7.0)
        % do not attempt to compile CHOLMOD with large file support
        include = [include ' -DNLARGEFILE'] ;
    elseif (~pc)
        % Linux/Unix require these flags for large file support
        include = [include ' -D_FILE_OFFSET_BITS=64 -D_LARGEFILE64_SOURCE'] ;
    end

    if (v < 6.5)
        % logical class does not exist in MATLAB 6.1 or earlie
        include = [include ' -DMATLAB6p1_OR_EARLIER'] ;
    end

    % Determine the METIS path, and whether or not METIS is available
% $$$ if (nargin == 0)
    metis_path = '../../metis-4.0' ;
% $$$ end
% $$$ if (strcmp (metis_path, 'no metis'))
% $$$     metis_path = '' ;
% $$$ end
    have_metis = (~isempty (metis_path)) ;

    % fix the METIS 4.0.1 rename.h file
    if (have_metis)
        fprintf ('Compiling CHOLMOD with METIS on MATLAB Version %g\n', v) ;
        f = fopen ('rename.h', 'w') ;
        if (f == -1)
            error ('unable to create rename.h in current directory') ;
        end
        fprintf (f, '/* do not edit this file; generated by cholmod_make */\n') ;
        fprintf (f, '#undef log2\n') ;
        fprintf (f, '#include "%s/Lib/rename.h"\n', metis_path) ;
        fprintf (f, '#undef log2\n') ;
        fprintf (f, '#define log2 METIS__log2\n') ;
        fprintf (f, '#include "mex.h"\n') ;
        fprintf (f, '#define malloc mxMalloc\n') ;
        fprintf (f, '#define free mxFree\n') ;
        fprintf (f, '#define calloc mxCalloc\n') ;
        fprintf (f, '#define realloc mxRealloc\n') ;
        fclose (f) ;
        include = [include ' -I' metis_path '/Lib'] ;
    else
        fprintf ('Compiling CHOLMOD without METIS on MATLAB Version %g\n', v) ;
        include = ['-DNPARTITION ' include] ;
    end


    %-------------------------------------------------------------------------------
    % BLAS option
    %-------------------------------------------------------------------------------

    % This is exceedingly ugly.  The MATLAB mex command needs to be told where to
    % fine the LAPACK and BLAS libraries, which is a real portability nightmare.
    if (pc)
        if (v < 6.5)
            % MATLAB 6.1 and earlier: use the version supplied here
            lapack = 'lcc_lib/libmwlapack.lib' ;
        elseif (v < 7.5)
            lapack = 'libmwlapack.lib' ;
        else
            lapack = 'libmwlapack.lib libmwblas.lib' ;
            % There seems to be something weird how Matlab forms the paths
            % to lapack in Windows. If the above does not work try the
            % below by changing the path to your own Matlab directory.
            %lapack = 'C:\Program'' Files''\MATLAB\R2010a\extern\lib\win64\microsoft\libmwlapack.lib C:\Program'' Files''\MATLAB\R2010a\extern\lib\win64\microsoft\libmwblas.lib';
        end
    else
        if (v < 7.5)
            lapack = '-lmwlapack' ;
        else
            lapack = '-lmwlapack -lmwblas' ;
        end
    end

    %-------------------------------------------------------------------------------

    cholmod_path = [suiteSparse 'CHOLMOD/'];
    include = strrep(include, '../../', suiteSparse);
    include = strrep(include, '../', cholmod_path);
    include = strrep (include, '/', filesep) ;

    amd_src = { ...
        '../../AMD/Source/amd_1', ...
        '../../AMD/Source/amd_2', ...
        '../../AMD/Source/amd_aat', ...
        '../../AMD/Source/amd_control', ...
        '../../AMD/Source/amd_defaults', ...
        '../../AMD/Source/amd_dump', ...
        '../../AMD/Source/amd_global', ...
        '../../AMD/Source/amd_info', ...
        '../../AMD/Source/amd_order', ...
        '../../AMD/Source/amd_postorder', ...
        '../../AMD/Source/amd_post_tree', ...
        '../../AMD/Source/amd_preprocess', ...
        '../../AMD/Source/amd_valid' } ;

    camd_src = { ...
        '../../CAMD/Source/camd_1', ...
        '../../CAMD/Source/camd_2', ...
        '../../CAMD/Source/camd_aat', ...
        '../../CAMD/Source/camd_control', ...
        '../../CAMD/Source/camd_defaults', ...
        '../../CAMD/Source/camd_dump', ...
        '../../CAMD/Source/camd_global', ...
        '../../CAMD/Source/camd_info', ...
        '../../CAMD/Source/camd_order', ...
        '../../CAMD/Source/camd_postorder', ...
        '../../CAMD/Source/camd_preprocess', ...
        '../../CAMD/Source/camd_valid' } ;

    colamd_src = {
        '../../COLAMD/Source/colamd', ...
        '../../COLAMD/Source/colamd_global' } ;

    ccolamd_src = {
        '../../CCOLAMD/Source/ccolamd', ...
        '../../CCOLAMD/Source/ccolamd_global' } ;

    metis_src = {
        'Lib/balance', ...
        'Lib/bucketsort', ...
        'Lib/ccgraph', ...
        'Lib/coarsen', ...
        'Lib/compress', ...
        'Lib/debug', ...
        'Lib/estmem', ...
        'Lib/fm', ...
        'Lib/fortran', ...
        'Lib/frename', ...
        'Lib/graph', ...
        'Lib/initpart', ...
        'Lib/kmetis', ...
        'Lib/kvmetis', ...
        'Lib/kwayfm', ...
        'Lib/kwayrefine', ...
        'Lib/kwayvolfm', ...
        'Lib/kwayvolrefine', ...
        'Lib/match', ...
        'Lib/mbalance2', ...
        'Lib/mbalance', ...
        'Lib/mcoarsen', ...
        'Lib/memory', ...
        'Lib/mesh', ...
        'Lib/meshpart', ...
        'Lib/mfm2', ...
        'Lib/mfm', ...
        'Lib/mincover', ...
        'Lib/minitpart2', ...
        'Lib/minitpart', ...
        'Lib/mkmetis', ...
        'Lib/mkwayfmh', ...
        'Lib/mkwayrefine', ...
        'Lib/mmatch', ...
        'Lib/mmd', ...
        'Lib/mpmetis', ...
        'Lib/mrefine2', ...
        'Lib/mrefine', ...
        'Lib/mutil', ...
        'Lib/myqsort', ...
        'Lib/ometis', ...
        'Lib/parmetis', ...
        'Lib/pmetis', ...
        'Lib/pqueue', ...
        'Lib/refine', ...
        'Lib/separator', ...
        'Lib/sfm', ...
        'Lib/srefine', ...
        'Lib/stat', ...
        'Lib/subdomains', ...
        'Lib/timing', ...
        'Lib/util' } ;
    
    for i = 1:length (metis_src)
        metis_src {i} = [metis_path '/' metis_src{i}] ;
    end

    cholmod_matlab = { '../MATLAB/cholmod_matlab' } ;

    cholmod_src = {
        '../Core/cholmod_aat', ...
        '../Core/cholmod_add', ...
        '../Core/cholmod_band', ...
        '../Core/cholmod_change_factor', ...
        '../Core/cholmod_common', ...
        '../Core/cholmod_complex', ...
        '../Core/cholmod_copy', ...
        '../Core/cholmod_dense', ...
        '../Core/cholmod_error', ...
        '../Core/cholmod_factor', ...
        '../Core/cholmod_memory', ...
        '../Core/cholmod_sparse', ...
        '../Core/cholmod_transpose', ...
        '../Core/cholmod_triplet', ...
        '../Check/cholmod_check', ...
        '../Check/cholmod_read', ...
        '../Check/cholmod_write', ...
        '../Cholesky/cholmod_amd', ...
        '../Cholesky/cholmod_analyze', ...
        '../Cholesky/cholmod_colamd', ...
        '../Cholesky/cholmod_etree', ...
        '../Cholesky/cholmod_factorize', ...
        '../Cholesky/cholmod_postorder', ...
        '../Cholesky/cholmod_rcond', ...
        '../Cholesky/cholmod_resymbol', ...
        '../Cholesky/cholmod_rowcolcounts', ...
        '../Cholesky/cholmod_rowfac', ...
        '../Cholesky/cholmod_solve', ...
        '../Cholesky/cholmod_spsolve', ...
        '../MatrixOps/cholmod_drop', ...
        '../MatrixOps/cholmod_horzcat', ...
        '../MatrixOps/cholmod_norm', ...
        '../MatrixOps/cholmod_scale', ...
        '../MatrixOps/cholmod_sdmult', ...
        '../MatrixOps/cholmod_ssmult', ...
        '../MatrixOps/cholmod_submatrix', ...
        '../MatrixOps/cholmod_vertcat', ...
        '../MatrixOps/cholmod_symmetry', ...
        '../Modify/cholmod_rowadd', ...
        '../Modify/cholmod_rowdel', ...
        '../Modify/cholmod_updown', ...
        '../Supernodal/cholmod_super_numeric', ...
        '../Supernodal/cholmod_super_solve', ...
        '../Supernodal/cholmod_super_symbolic', ...
        '../Partition/cholmod_ccolamd', ...
        '../Partition/cholmod_csymamd', ...
        '../Partition/cholmod_camd', ...
        '../Partition/cholmod_metis', ...
        '../Partition/cholmod_nesdis' } ;

    if (pc)
        % Windows does not have drand48 and srand48, required by METIS.  Use
        % drand48 and srand48 in CHOLMOD/MATLAB/Windows/rand48.c instead.
        obj_extension = '.obj' ;
        cholmod_matlab = [cholmod_matlab {[cholmod_path 'MATLAB\Windows\rand48']}] ;
        include = [include ' -I' cholmod_path '\MATLAB\Windows'] ;
    else
        obj_extension = '.o' ;
    end

    % compile each library source file
    obj = '' ;

    source = [amd_src colamd_src ccolamd_src camd_src cholmod_src cholmod_matlab] ;
    if (have_metis)
        source = [metis_src source] ;
    end
    if exist('OCTAVE_VERSION', 'builtin')
      for i1=1:length(source)
        source{i1} = strrep(source{i1}, '../../', suiteSparse);
        source{i1} = strrep(source{i1}, '../', cholmod_path);
      end
    else
      source = strrep(source, '../../', suiteSparse);
      source = strrep(source, '../', cholmod_path);
    end

    kk = 0 ;
    
    for f = source
        ff = strrep (f {1}, '/', filesep) ;
        slash = strfind (ff, filesep) ;
        if (isempty (slash))
            slash = 1 ;
        else
            slash = slash (end) + 1 ;
        end
        o = ff (slash:end) ;
        obj = [obj  ' ' o obj_extension] ;					    %#ok
        if ~exist('OCTAVE_VERSION','builtin')
          s = sprintf ('mex %s -DDLONG -O %s -c %s.c', d, include, ff) ;
        else
          s = sprintf ('mex %s -DDLONG %s -c %s.c', d, include, ff) ;
        end
        kk = do_cmd (s, kk, details) ;
    end
    
    if pc
        % compile mexFunctions
        mex_src =  'winCsource\spinv';
        outpath = 'private\spinv';
        if ~exist('OCTAVE_VERSION','builtin')
          s = sprintf ('mex %s -DDLONG -O %s -output %s %s.c', d, include, outpath, mex_src) ;
        else
          s = sprintf ('mex %s -DDLONG %s -output %s %s.c', d, include, outpath, mex_src) ;
        end
        s = [s obj];
        s = [s ' '];
        s = [s lapack];
        kk = do_cmd (s, kk, details) ;
        
        %mex_src = 'linuxCsource/ldlrowupdate';
        mex_src = 'winCsource\ldlrowupdate';
        outpath = 'private\ldlrowupdate';
        if ~exist('OCTAVE_VERSION','builtin')
          s = sprintf ('mex %s -DDLONG -O %s -output %s %s.c', d, include, outpath, mex_src) ;
        else
          s = sprintf ('mex %s -DDLONG %s -output %s %s.c', d, include, outpath, mex_src) ;
        end
        s = [s obj];
        s = [s ' '];
        s = [s lapack];
        kk = do_cmd (s, kk, details) ;
    else
        % compile mexFunctions
        mex_src =  'linuxCsource/spinv';
        outpath = 'private/spinv';
        if ~exist('OCTAVE_VERSION','builtin')
          s = sprintf ('mex %s -DDLONG -O %s -output %s %s.c', d, include, outpath, mex_src) ;
        else
          s = sprintf ('mex %s -DDLONG %s -output %s %s.c', d, include, outpath, mex_src) ;
        end
        s = [s obj];
        s = [s ' '];
        s = [s lapack];
        kk = do_cmd (s, kk, details) ;
        
        %mex_src = 'linuxCsource/ldlrowupdate';
        mex_src = 'linuxCsource/ldlrowupdate';
        outpath = 'private/ldlrowupdate';
        if ~exist('OCTAVE_VERSION','builtin')
          s = sprintf ('mex %s -DDLONG -O %s -output %s %s.c', d, include, outpath, mex_src) ;
        else
          s = sprintf ('mex %s -DDLONG %s -output %s %s.c', d, include, outpath, mex_src) ;
        end
        s = [s obj];
        s = [s ' '];
        s = [s lapack];
        kk = do_cmd (s, kk, details) ;
    end
    % clean up
    s = ['delete ' obj] ;
    
    do_cmd (s, kk, details) ;
    fprintf ('\nGP package succesfully compiled \n') ;
end

%-------------------------------------------------------------------------------
function kk = do_cmd (s, kk, details)
%DO_CMD: evaluate a command, and either print it or print a "."
if (details)
    fprintf ('%s\n', s) ;
else
    if (mod (kk, 60) == 0)
	fprintf ('\n') ;
    end
    kk = kk + 1 ;
    fprintf ('.') ;
end
if ~exist('OCTAVE_VERSION','builtin')
  eval (s) ;
else
  tmp=regexp(s,'\S+','match');
  if strcmp(tmp{1},'delete')
    for i=1:length(tmp)-1
      eval([tmp{1} ' ' tmp{i+1}]);
    end
  else
    eval (s) ;
  end
end

%-------------------------------------------------------------------------------
function v = getversion
% determine the MATLAB version, and return it as a double.
v = sscanf (version, '%d.%d.%d') ;
v = 10.^(0:-1:-(length(v)-1)) * v ;
