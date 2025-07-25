function prevstream = setrandstream(seed, stream)
%SETRANDSTREAM Set random stream
%
%  Description
%    SETRANDSTREAM is a compatibility function which calls appropriate
%    functions depdening on the Matlab version or Octave (Matlab has
%    changed random number syntax three times during the existence of
%    GPstuff and Octave naturally has its own syntax).
%
%    CURRSTREAM = SETRANDSTREAM()
%    Return current random stream as a random stream object CURRSTREAM.
%
%    PREVSTREAM = SETRANDSTREAM(SEED)
%    Set MATLAB random stream to default stream (Mersenne Twister) with 
%    state/seed SEED and return previous stream PREVSTREAM. This function 
%    takes into account used MATLAB version, and uses correct function 
%    based on that. 
%    
%    PREVSTREAM = SETRANDSTREAM(SEED, STREAM)
%    Set MATLAB random stream to STREAM, with state/seed SEED. Here STREAM
%    is string defining random number generator, e.g. 'mt19937ar' or 
%    'twister' (in new MATLAB versions) for Mersenne Twister.
%
%    PREVSTREAM = SETRANDSTREAM(STREAMOBJ)
%    Set MATLAB random stream to STREAMOBJ. Here STREAMOBJ is random stream
%    object returned from e.g. this function or rng function in new MATLAB
%    versions.
%
%  See also
%    RANDSTREAM, RNG
%   
% Copyright (c) 2012, Ville Tolvanen

% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.
  
p=which('randn');
p=strfind(p, 'RcppOctave');
% Check if RcppOctave is active as the seed initialization does not work
% in RcppOctave
if isempty(p)
  if nargin>=1 && ~isnumeric(seed)
    % First argument is random stream object
    stream=seed;
    if str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      prevstream = RandStream.setDefaultStream(stream);
    else
      prevstream=rng(stream);
    end
  else
    if nargin<2
      if nargin<1
        % Get current random stream
        if exist('OCTAVE_VERSION', 'builtin')
          prevstream(1) = randn('seed');
          prevstream(2) = rand('seed');
        elseif str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
          prevstream = RandStream.getDefaultStream();
        else
          prevstream = rng;
        end
        return
      else
        % If stream is not provided, use Mersenne Twister
        stream='mt19937ar';
      end
    end
    if isempty(seed)
      % Default seed
      seed=0;
    end
    if exist('OCTAVE_VERSION', 'builtin')
      prevstream(1) = randn('seed');
      prevstream(2) = rand('seed');
      if length(seed)==1
        seed(2)=seed(1);
      end
      randn('seed', seed(1));
      rand('seed', seed(2));
    elseif str2double(regexprep(version('-release'), '[a-c]', '')) < 2012
      if ischar(stream)
        stream = RandStream(stream,'Seed',seed);
      end
      prevstream = RandStream.setDefaultStream(stream);
    else
      if ischar(stream)
        prevstream = rng(seed,stream);
      else
        prevstream=rng(stream);
      end
    end
  end
else
  prevstream=[];
end

end

