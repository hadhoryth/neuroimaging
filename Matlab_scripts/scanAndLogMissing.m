function scanAndLogMissing(root, csv_data)
% info - struct with locations of rearanged folders
% fiels: name - root folder name; path - location
%        ad_name - folder for AD name; ad_path - location
%        normal_name - folder for Normal patients name; normal_path - location

fprintf('Creating logfile ..................');
logs_path = mfullfile(root,'logs.txt');
fid = fopen(logs_path, 'w');
fprintf('OK\n');
fprintf(fid, 'Patient,   Missing\n');

tree = dir(root);
tree = checkHiddenFolders({tree.name}, 1);
full_tree = mfullfile(root, tree);
sub_trees = checkHiddenFolders(full_tree);

for i = 1 : length(sub_trees)
    dx_data= findDXData('', csv_data, full_tree{i});    
    st = struct('pet', 'Missing', 'fdg', 'Missing', 'mri', 'Missing');
    if(length(sub_trees{i}) == 3)
        st.pet = 'OK'; st.fdg = 'OK'; st.mri = 'OK';
        printStatus(tree{i}, st.pet, st.fdg, st.mri, dx_data.dx_change);
        startPreprocess(full_tree{i}, sub_trees{i}, dx_data.dx_change, '', dx_data);
    else
        parts = checkWhatMissing(sub_trees{i});
        printStatus(tree{i}, st.pet, st.fdg, st.mri, dx_data.dx_change);
        if(isempty(strfind([parts{:}], 'MRI')) == 1 && isempty(strfind([parts{:}], 'PET')) == 1)
            startPreprocess(full_tree{i}, sub_trees{i}, dx_data.dx_change, parts, dx_data);
        end
    end
end

fclose(fid); 
        
    function toLogs = checkWhatMissing(loc)
        toLogs = {'PET,', 'FDG,', 'MRI'};
        for ii = 1 : length(loc)
            if(isempty(strfind(loc{ii}, 'AV45')) == 0)
                st.pet = 'OK';
                toLogs{1} = {};
            elseif(isempty(strfind(loc{ii}, 'MT1')) == 0)
                st.mri = 'OK';
                toLogs{3} = {};
            else
                st.fdg = 'OK';
                toLogs{2} = {};
            end
        end
        toLogs = toLogs(~cellfun('isempty',toLogs));
        fprintf(fid,'%s, %s\n', tree{i}, cell2mat(toLogs));
    end

    function printStatus(patient, pet, fdg, mri, mode)
        fprintf('Patient %s(%s) ...... ', patient, mode)
        fprintf('PET - ');
        cprintf('*comment', pet);
        fprintf(', FDG - ');
        cprintf('*comment', fdg);
        fprintf(', MRI - ');
        cprintf('*comment', mri);
        fprintf('\n');
    end

    function startPreprocess(path, subpath, mode, missing, dx_data)
        fprintf('Preprocessing images: ......... ');
        f_locs = mfullfile(path, subpath);               
        if(isempty(strfind(missing, 'FDG')) == 0)            
            [mri, fdg] = deal(f_locs{2}, cell(1,1));
        else
            [mri, fdg] = deal(f_locs{3}, f_locs{2});            
        end
        
        f = struct('pet', f_locs{1},'fdg', fdg, 'mri', mri);
        preprocessImages(f, mode, dx_data);
        fprintf('OK\n');
    end

end