function removeMRIFolder(items, ~)
% items - cell array 
if(nargin < 2 && isempty(items) == 0)
    cprintf('comment', '\n----------Removing unused MRI data--------------\n');
end
    while ~isempty(items)           
        if(length(items{end}) > 1 && iscell(items{end}) == 1)
            removeMRIFolder(items{end}, 1);
            items{end} = {};
            items = items(~cellfun('isempty', items));
            continue;
        end
        folder = char(items{end});
        path = fileparts(folder);
        if(isdir(path) == 1)
           rmdir(path, 's');
        end        
        items{end} = {};
        items = items(~cellfun('isempty', items));
    end
end