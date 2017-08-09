function rearrangePreFolders(flds, csv_data)
    %Normal
    [tree_n, tree_nm] = deal(checkHiddenFolders({flds.normal_path}), checkHiddenFolders({flds.normal_meta_path}));
    [tree_n, tree_nm] = deal(tree_n{1}, tree_nm{1}); 
    %AD
    [tree_ad, tree_adm] = deal(checkHiddenFolders({flds.ad_path}), checkHiddenFolders({flds.ad_meta_path}));
    [tree_ad, tree_adm] = deal(tree_ad{1}, tree_adm{1}); 
    
    [what_where, k] = deal({}, 1); 
    
    %% Processing normal and normal_meta data 
    scan(tree_n, flds.normal_path, 'normal', 0);
    scan(tree_nm, flds.normal_meta_path, 'normal_meta', 1);
    
    %% Processing ad and ad_meta data
    scan(tree_ad, flds.ad_path, 'ad', 0);
    scan(tree_adm, flds.ad_meta_path, 'ad_meta', 1);
    
    %% Rearranging
    i = length(what_where);
    while i >= 1
        if(~strcmpi(what_where{i}{1}, what_where{i}{2}))            
            if(exist(what_where{i}{2}, 'dir') == 7)
                rmdir(what_where{i}{2}, 's');
            end
            fprintf('Moving from %s to %s\n', what_where{i}{1}, what_where{i}{2});
            movefile(what_where{i}{1}, what_where{i}{2});
        end
        i = i - 1;
    end    
    
    function scan(tree, path, current_mode, is_meta)
        for ii = 1 : length(tree)
            var = findDXData(tree{ii}, csv_data);
            if(isstruct(var) == 1)
                if((strcmpi(current_mode, var.dx_change) == 0 && is_meta == 0) || ...
                   (strcmpi(current_mode, strcat(var.dx_change, '_meta')) == 0 && is_meta == 1))
                    out_path = f_getPath(flds, var.dx_change, current_mode)
                    what_where{k}{1} = mfullfile(path, tree{ii});
                    what_where{k}{2} = mfullfile(out_path, tree{ii});
                    k = k + 1;
                end
            end
        end
    end
end