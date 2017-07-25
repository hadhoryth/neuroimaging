function patient_ids = csv_rearrange(home, xls_files, mat_files)   
    % .mat files have to have variables
    % !During tranformation csv -> xlsx keep the EXAM_DATE column as a text!
    % data_ids - where all information is stored (RID, Patient ID, Gender)
    % data_bl_change - where RID and DX_Change

    
    if(exist(mat_files{1}, 'file') == 2)
        load(mat_files{1});
        load(mat_files{2});
    else
        data_ids = csv2struct(xls_files{1});
        save(strcat(home,'data_ids.mat'), 'data_ids');
        data_bl_change = csv2struct(xls_files{2});
        save(strcat(home,'data_bl_change.mat'), 'data_bl_change');
    end

    % Get all available DX_Change
    [~, idx] = find(isnan([data_bl_change.DXCHANGE{:}]) == 0);
    rids = data_bl_change.RID(idx);
    dx_change = cell2mat(data_bl_change.DXCHANGE(idx));
    dates = data_bl_change.EXAMDATE(idx);

    % Get info based on RID and
    rids_length = length(idx);
    patient_ids = cell(rids_length,1);     
   
    for i = 1 : rids_length
        idx = find(ismember(data_ids.RID, rids(i)) > 0);
        [ref_date, other] = deal(date2num(dates(i)), date2num(data_ids.EXAMDATE(idx)));
        date_idx = pickValidDate(ref_date.year, ref_date.month, other);
        if(length(date_idx) > 1)
            date_idx = date_idx(1);
        elseif(isempty(date_idx) == 1)
            continue;
        end 
        idx = idx(date_idx);
        patient_ids{i} = struct('id', data_ids.PTID(idx), 'age', data_ids.AGE(idx),...
            'gender', data_ids.PTGENDER(idx), 'mmse', data_ids.MMSE(idx), ...
            'dx_change', dxchange2label(dx_change(i)));       
    end
    
    patient_ids = patient_ids(~cellfun('isempty', patient_ids));
    
    function mark = dxchange2label(dx_change)
        mark = '';
        switch(dx_change)
            case {1, 7, 9}
               mark = 'Normal';
            case {2, 4, 8}
                mark = 'MCI';
            case {3, 5, 6}
                mark = 'AD';
        end
    end
end
