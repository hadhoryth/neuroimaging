function tree = rearangeFolders(root_dir, out_dir)
root_content = checkHiddenFolders({root_dir});
root_content = root_content{1};

% Creating output directory
output = struct('name', 'ADNI_Rearanged');
output.path = fullfile(out_dir, output.name)

if(exist(output.path, 'dir') == 0)
    fprintf('Creating output directory ....... ');
    mkdir (output.name);
    cprintf('*comment', 'Ok\n');
end

% Main loop through root
for i = 1 : length(root_content)
    if(isempty(strfind(lower(root_content{i}), 'normal')) == 0)        
        if(isempty(strfind(lower(root_content{i}), 'metadata')) == 0)
            createOutputDirs('normal_meta');
            handleFolderCoping(root_content{i}, 'normal_meta', output.normal_meta_path);
        else
           createOutputDirs('normal');
           handleFolderCoping(root_content{i}, 'normal', output.normal_path);
        end
    elseif(isempty(strfind(lower(root_content{i}), 'ad')) == 0)        
        if(isempty(strfind(lower(root_content{i}), 'metadata')) == 0)
            createOutputDirs('ad_meta');
            handleFolderCoping(root_content{i}, 'ad_meta', output.ad_meta_path);
        else
            createOutputDirs('ad');
            handleFolderCoping(root_content{i}, 'ad', output.ad_path);
        end
    end
    
end

tree = output;

    function handleFolderCoping(fld_path, mode, output_mode)
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
   
    function createOutputDirs(mode)
        if(strcmp(mode, 'normal'))
            if (isfield(output, 'normal') == 0)               
                output.normal = 'Normal';
                output.normal_path = strcat(output.path, '/', output.normal);
                doCreate(output.normal_path, output.normal);                
            end            
        elseif (strcmp(mode, 'normal_meta'))
            if (isfield(output, 'normal_meta') == 0)               
                output.normal_meta = 'Normal_meta';
                output.normal_meta_path = strcat(output.path, '/', output.normal_meta);
                doCreate(output.normal_meta_path, output.normal_meta);                
            end
        elseif (strcmp(mode, 'ad_meta'))
            if (isfield(output, 'ad_meta') == 0)               
                output.ad_meta = 'AD_meta';
                output.ad_meta_path = strcat(output.path, '/', output.ad_meta);
                doCreate(output.ad_meta_path, output.ad_meta);                
            end
        else
            if (isfield(output, 'ad') == 0)                
                output.ad = 'AD';
                output.ad_path = strcat(output.path, '/', output.ad);
                doCreate(output.ad_path, output.ad);
            end
        end
        
        function doCreate(dir_loc, mode)
            if (exist(dir_loc, 'dir') == 0)
                fprintf('Creating directory for %s patients ....... ', mode);
                mkdir(dir_loc);
                cprintf('*comment', 'Ok\n');
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