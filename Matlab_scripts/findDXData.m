function data = findDXData(ptid, dx_data, fullpath)
% dx_data - cell array of structs
% where struct has field: id 

global data_ids


if(nargin > 2)
    [~, ptid] = fileparts(fullpath);
end

data = struct('id', [], 'age', [], 'mmse', [], 'dx_change', [], 'gender', []);
mmse = 0;
k = 1;
    for i = 1 : length(dx_data)
       if(strcmp(dx_data{i}.id, ptid) == 1 && isnan(dx_data{i}.mmse) == 0)
           data = dx_data{i}; 
           data.dx_change = getNormDX(data.dx_change);
           mmse = mmse + (dx_data{i}.mmse - mmse)/k;
           k = k + 1;
       end
    end
    if(isempty(data.id) == 1) 
        fprintf('DX_change data for patient %s not found checking general list\n', ptid);
        getWithOutDX(ptid);        
    end
    
    
    function getWithOutDX(ptid)        
        ii = find(strcmp(data_ids.PTID, ptid) == 1);
        if(isempty(ii))
            return;
        end
        data.id = ptid;
        data.age = mean(data_ids.AGE(ii));
        
        mmse = cell2mat(data_ids.MMSE(ii));
        mmse(isnan(mmse)) = [];
        data.mmse = mean(mmse);
        
        dx_change = data_ids.DX_bl(ii);
        data.dx_change = getNormDX(char(dx_change{1})); 
        
        pt_gender = data_ids.PTGENDER(ii);
        data.gender = pt_gender{1};
    end

    function dx_norm = getNormDX(dx)
        dx_norm = 'Normal';
        if(strcmpi(dx, 'mci'))
            dx_norm = 'MCI';
        elseif(strcmpi(dx, 'ad'))
            dx_norm = 'AD';
        end        
    end

end