function data = findDXData(ptid, dx_data, fullpath)
% dx_data - cell array of structs
% where struct has field: id 
if(nargin > 2)
    [~, ptid] = fileparts(fullpath);
end

data = NaN;
    for i = 1 : length(dx_data)
       if(strcmp(dx_data{i}.id, ptid) == 1)
           data = dx_data{i};
           return
       end
   end
end