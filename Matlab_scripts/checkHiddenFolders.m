function [folders, sizes] = checkHiddenFolders(loc, ~)
    if (nargin > 1)
        folders = removeHidden(loc);
        return;
    end
    folders = cell(1, length(loc));
    sizes = zeros(length(loc), 2);
    for ii = 1 : length(loc)           
       folder = dir(loc{ii});
       subfolders = {folder.name};
%        k = 1;
%        tmp = {};
%        for jj = 1 : length(subfolders)
%           if(strcmp(subfolders{jj}, '.') ~= 1 && strcmp(subfolders{jj}, '..') ~= 1 &&...
%              strcmp(subfolders{jj}, '.DS_Store') ~= 1)
%              tmp{k} = char(subfolders{jj});
%              k = k + 1;
%           end
%        end
       tmp = removeHidden(subfolders);
       folders{ii} = tmp;
       sizes(ii,1) = length(tmp);
       sizes(ii,2) = ii;
    end
    
    function tmp = removeHidden(path)
       k = 1;
       tmp = {};
       for jj = 1 : length(path)
          if(strcmp(path{jj}, '.') ~= 1 && strcmp(path{jj}, '..') ~= 1 &&...
             strcmp(path{jj}, '.DS_Store') ~= 1)
             tmp{k} = char(path{jj});
             k = k + 1;
          end
       end
    end
end
