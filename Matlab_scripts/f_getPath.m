function path = f_getPath(info, dx_change, mode)
    path = '';
    if(strcmpi(dx_change, 'normal'))
        if(isempty(regexpi(mode,'_meta$')) == 0)
            path = info.normal_meta_path;
            return;
        end
        path = info.normal_path;
    elseif(strcmpi(dx_change, 'ad'))
        if(isempty(regexpi(mode,'_meta$')) == 0)
            path = info.ad_meta_path;
            return;
        end
        path = info.ad_path;       
    end
end