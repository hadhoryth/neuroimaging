function f = fullfile(varargin)
%FULLFILE Build full file name from parts.
%   F = fullfile(FOLDERNAME1, FOLDERNAME2, ..., FILENAME) builds a full
%   file specification F from the folders and file name specified. Input
%   arguments FOLDERNAME1, FOLDERNAME2, etc. and FILENAME can be strings,
%   a scalar cell string, or same-sized cell arrays of strings. The output
%   of fullfile is conceptually equivalent to
%
%      F = [FOLDERNAME1 filesep FOLDERNAME2 filesep ... filesep FILENAME]
%
%   except that care is taken to handle the cases when the folders begin or
%   end with a file separator.
%
%   Examples
%     % To build platform dependent paths to files:
%        fullfile(matlabroot,'toolbox','matlab','general','Contents.m')
%
%     % To build platform dependent paths to a folder:
%        fullfile(matlabroot,'toolbox','matlab',filesep)
%
%     % To build a collection of platform dependent paths to files:
%        fullfile(toolboxdir('matlab'),'iofun',{'filesep.m';'fullfile.m'})
%
%   See also FILESEP, PATHSEP, FILEPARTS.

%   Copyright 1984-2014 The MathWorks, Inc.
    
    narginchk(1, Inf);
    persistent fileSeparator;
    if isempty(fileSeparator)
        fileSeparator = filesep;
    end
    argIsACell = cellfun('isclass', varargin, 'cell');
    theInputs = varargin;
    f = theInputs{1};
    try
        if nargin == 1
            f = refinePath(f, fileSeparator);
            return;
        elseif any(argIsACell)
            theInputs(cellfun(@(x)~iscell(x)&&isempty(x), theInputs)) = [];
        else
            theInputs(cellfun('isempty', theInputs)) = '';
        end
        
        if length(theInputs)>1
            theInputs{1} = ensureTrailingFilesep(theInputs{1}, fileSeparator);
        end
        if ~isempty(theInputs)
            theInputs(2,:) = {fileSeparator};
            theInputs{2,1} = '';
            theInputs(end) = '';
            if any(argIsACell)
                f = strcat(theInputs{:});
            else
                f = [theInputs{:}];
            end
        end
        f = refinePath(f,fileSeparator);
    catch exc
        locHandleError(exc, theInputs(1,:));
    end
end


function f = ensureTrailingFilesep(f,fileSeparator)
    if iscell(f)
        for i=1:numel(f)
            f{i} = addTrailingFileSep(f{i},fileSeparator);
        end
    else
        f = addTrailingFileSep(f,fileSeparator);
    end
end

function str = addTrailingFileSep(str, fileSeparator)
    persistent bIsPC
    if isempty (bIsPC)
        bIsPC = ispc;
    end
    if ~isempty(str) && (str(end) ~= fileSeparator && ~(bIsPC && str(end) == '/'))
        str = [str, fileSeparator];
    end
end
function f = refinePath(f, fs)
    persistent singleDotPattern multipleFileSepPattern
       
    if isempty(singleDotPattern)
        singleDotPattern = [fs, '.', fs];
        multipleFileSepPattern = [fs, fs];
    end   
    
    f = strrep(f, '/', fs);
          
    if iscell(f)        
        hasSingleDotCell = ~cellfun('isempty',strfind(f, singleDotPattern));
        if any(hasSingleDotCell)
            f(hasSingleDotCell) = replaceSingleDots(f(hasSingleDotCell), fs);
        end
        
        hasMultipleFileSepCell = ~cellfun('isempty',strfind(f, multipleFileSepPattern));
        if any(hasMultipleFileSepCell)
            f(hasMultipleFileSepCell) = replaceMultipleFileSeps(f(hasMultipleFileSepCell), fs);
        end
    else
        if ~isempty(strfind(f, singleDotPattern))
            f = replaceSingleDots(f, fs);
        end
        
        if ~isempty(strfind(f, multipleFileSepPattern))
            f = replaceMultipleFileSeps(f, fs);
        end          
    end
end

function f = replaceMultipleFileSeps(f, fs)    
    persistent fsEscape multipleFileSepRegexpPattern
    if isempty(fsEscape)
        fsEscape = ['\', fs];
        if ispc
            multipleFileSepRegexpPattern = '(^(\\\\\?\\.*|(\w+:)?\\+))|(\\)\\+';
        else
            multipleFileSepRegexpPattern = ['(?<!^(\w+:)?' fsEscape '*)(', fsEscape, ')', fsEscape '+'];
        end
    end
    f = regexprep(f, multipleFileSepRegexpPattern, '$1');
end

function f = replaceSingleDots(f, fs)   
    persistent fsEscape singleDotRegexpPattern
    if isempty(fsEscape)
        fsEscape = ['\', fs];
        if ispc
            singleDotRegexpPattern = '(^\\\\(\?\\.*|\.(?=\\)))|(\\)(?:\.\\)+';
        else
            singleDotRegexpPattern = ['(',fsEscape,')', '(?:\.', fsEscape, ')+'];
        end
    end
    f = regexprep(f, singleDotRegexpPattern, '$1');
end

function locHandleError(theException, theInputs)
    idToThrow = 'MATLAB:fullfile:UnknownError';
    switch theException.identifier
    case {'MATLAB:catenate:dimensionMismatch', ...
            'MATLAB:strcat:InvalidInputSize', 'MATLAB:strrep:MultiRowInput', 'MATLAB:strcat:NumberOfInputRows', 'MATLAB:UnableToConvert'}
        % Verify that all char inputs have only one row and that the sizes
        % of all nonscalar cell inputs match.
        firstNonscalarCellArg = struct('idx', 0, 'size', 0);
        for argIdx = 1:numel(theInputs)
            currentArg = theInputs{argIdx};
            if isscalar(currentArg)
                continue;
            elseif ischar(currentArg) && ~isrow(currentArg)
                idToThrow = 'MATLAB:fullfile:NumCharRowsExceeded';
            elseif iscell(currentArg)
                currentArgSize = size(currentArg);
                if firstNonscalarCellArg.idx == 0
                    firstNonscalarCellArg.idx = argIdx;
                    firstNonscalarCellArg.size = currentArgSize;
                elseif ~isequal(currentArgSize, firstNonscalarCellArg.size)
                    idToThrow = 'MATLAB:fullfile:CellstrSizeMismatch';
                end
            end
        end
    otherwise
        % Verify that correct input types were specified.
        argIsInvalidType = ~cellfun(@(arg)isnumeric(arg)&&isreal(arg)||ischar(arg)||iscellstr(arg), theInputs);
        if any(argIsInvalidType)
            idToThrow = 'MATLAB:fullfile:InvalidInputType';
        end
    end
    excToThrow = MException(message(idToThrow));
    excToThrow = excToThrow.addCause(theException);
    throwAsCaller(excToThrow);
end