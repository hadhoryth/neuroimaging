function rearrangePreFolders(root, csv_data)  

global config_defaults data_ids
home = '/Users/XT/Documents/PhD/Granada/neuroimaging';
xls_home = mfullfile(home,'csv_dataset/');
config_defaults.mat_files = strcat(xls_home, {'data_ids.mat', 'data_bl_change.mat'})
data_ids = load (config_defaults.mat_files{1});
data_ids = data_ids.data_ids;

    root_content = checkHiddenFolders({root});
    root_content = root_content{1};
    

    [what_where, k] = deal({}, 1);
    for i = 1 : length(root_content)
        var = findDXData(root_content{i}, csv_data);
        if(~isempty(var.id))
            out_path = createOutputDirs(var.dx_change);
            what_where{k}{1} = mfullfile(root, root_content{i});
            what_where{k}{2} = out_path;         
            k = k + 1;      
        end
    end

    % Rearranging    
    while ~isempty(what_where)
%         if(~strcmpi(what_where{end}{1}, what_where{end}{2}))
%             if(exist(what_where{end}{2}, 'dir') == 7)
%                 rmdir(what_where{end}{2}, 's');
%             end
            fprintf('Moving from %s to %s\n', what_where{end}{1}, what_where{end}{2});
            movefile(what_where{end}{1}, what_where{end}{2});
            what_where{end} = {};
            what_where = what_where(~cellfun('isempty',what_where));
%         end        
    end
    
    function out = createOutputDirs(mode)    
        out = mfullfile(root, mode);          
        if (exist(out, 'dir') == 0)
            fprintf('Creating directory for %s patients ....... ', mode);
            mkdir(root, mode);
            cprintf('*comment', 'Ok\n');
        end
    end

end