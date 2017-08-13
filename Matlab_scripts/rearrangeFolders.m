function tree = rearrangeFolders(root_dir, out_dir)
% Use this function for non preproccessed images
root_content = checkHiddenFolders({root_dir});
root_content = root_content{1};

% Creating output directory for all patients folders
output = struct('name', 'ADNI_Rearranged');
output.path = mfullfile(out_dir, output.name);

if(exist(output.path, 'dir') == 0)
    fprintf('Creating output directory ....... ');
    mkdir (out_dir, output.name);
    cprintf('*comment', 'Ok\n');
end

% Main loop through root
modes = {'normal', 'ad', 'emci', 'lmci'};
for i = 1 : length(root_content)
    for j = 1 : length(modes)        
        s = startCoping(modes{j}, root_content{i}, output.path);
        if(s == 1) 
            break;  
        end
    end   
end
tree = output;

    function status = startCoping(mode, what, where)
       status = 0;
       name = strsplit(what, '_');
       if(strcmpi(name(end), mode))
           if(isempty(strfind(lower(what), 'metadata')) == 0)
               handleFolderCoping(what, where,strcat(mode, '_meta')); 
               return
           end
           handleFolderCoping(what, where, mode);
           status = 1;
       end
    end

    function handleFolderCoping(fld_path, output_mode, ~)
        % mode -> ad, normal
        % outputmode -> field of output struct        
        full_path = {mfullfile(root_dir, fld_path, '/ADNI')}; 
        subtree = checkHiddenFolders(full_path); 
        subtree = subtree{1};
        for ii = 1 : length(subtree)
            f_path = mfullfile(full_path{1}, subtree{ii});            
            o_path = mfullfile(output_mode, subtree{ii});            
            if(exist(o_path, 'dir') == 7)               
                f_path = strcat(f_path, '/');
                o_path = strcat(o_path, '/');            
            end              
            pp = checkHiddenFolders({f_path});
            pp = {mfullfile(o_path, pp{1})};
            if(exist(char(pp{1}), 'dir') == 0)                
                copyfile(f_path, o_path);
                performRearrangement(getDateFolder(pp{1}, 1));
            end
        end        
    end   

    function path = getDateFolder(path, ~)
       subtree = checkHiddenFolders(path);
       subtree = subtree{1};     
       path = strcat(path, '/', subtree);       
       if (nargin == 1)
           path = path{1};
       end      
    end

    function performRearrangement(foldersTree, trueRoot)
        for k = 1 : length(foldersTree)
            kk = 1; current = foldersTree(k);
            if(nargin > 1)
                trueCurrent = current;
            end
            deletionPath = cell(1,10);
            while (java.io.File(current).isFile() == 0)
                file = checkHiddenFolders(current);
                tmp = strcat(current, '/', file{1});
                if (length(file{1}) > 1 && isJustFiles(tmp) == 0)
                    performRearrangement(tmp, foldersTree(k));
                    break;
                end
                if(isempty(file{1}) == 1)
                    break;
                end
                
                current = strcat(current,'/',file{1});
                if(nargin > 1)
                    if (length(current) > 1 && isJustFiles(current) == 1)
                        for iii = 1 : length(current)
                            moveAndDeleteSubs(current{iii}, trueRoot, trueCurrent);
                        end
                        break;
                    end
                end
                
                if(length(current) > 1)
                    break;
                end
                if(length(file) == 1 && java.io.File(current).isFile() == 1 ...
                        && kk == 1 && nargin < 2)
                    break;
                end
                deletionPath{kk} = current;
                kk = kk + 1;
            end
            
            deletionPath = deletionPath(~cellfun('isempty', deletionPath));
            if (isempty(deletionPath) == 0)
                dest = foldersTree{k};
                dels = deletionPath(1);
                if(nargin > 1)
                    dest = trueRoot;
                    dels = trueCurrent;
                end
                moveAndDeleteSubs(deletionPath{end}, dest, dels);
            end
            
        end
    end

    function moveAndDeleteSubs(source, destination, toDelete)%
        printVar = char(source);
        cprintf('*blue', 'Moving file to the root .............. %s ',printVar(end-20:end));
        status = movefile(printVar, char(destination));
        if(status == 0)
            cprintf('red', 'Error during moving the file');
            fprintf('\n');
            return;
        else
            cprintf('red', ': OK\n');
            fprintf('\n');
        end
        rmdir(char(toDelete{1}),'s');
    end

    function status = isJustFiles(loc)
        status = 1;
        for ii = 1 :length(loc)
            if (java.io.File(loc{ii}).isFile() == 0)
                status = 0;
                break;
            end
        end
    end

    

end