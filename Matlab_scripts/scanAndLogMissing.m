function scanAndLogMissing(info, csv_data)
% info - struct with locations of rearanged folders
% fiels: name - root folder name; path - location
%        ad_name - folder for AD name; ad_path - location
%        normal_name - folder for Normal patients name; normal_path - location
fprintf('Creating logfile ..................');
logs_path = mfullfile(info.path,'logs.txt');
fid = fopen(logs_path, 'w');
fprintf('OK\n');
fprintf(fid, 'Patient,   Missing\n');

if(isempty(info.normal_path) == 0)
    fprintf('Normal patients data: \n');
    fprintf(fid, 'Normal patients data: \n');
    checkData(info.normal_path, 'Normal');
end

fprintf('\n');

if(isempty(info.ad_path) == 0)
    fprintf('AD patients data: \n');
    fprintf(fid, 'AD patients data: \n');
    checkData(info.ad_path, 'AD');
end

fclose(fid);

    function checkData(location, mode)
        % mode : 1 - Normal
        %        2 - AD
        tree = dir(location);
        tree = checkHiddenFolders({tree.name}, 1);        
        full_tree = mfullfile(location, tree);        
        sub_trees = checkHiddenFolders(full_tree);
        
        for i = 1 : length(sub_trees)
            dx_data= findDXData('', csv_data, full_tree{i});
            st = struct('pet', 'Missing', 'fdg', 'Missing', 'mri', 'Missing');
            if(length(sub_trees{i}) == 3)
                st.pet = 'OK'; st.fdg = 'OK'; st.mri = 'OK'; 
                printStatus(tree{i}, st.pet, st.fdg, st.mri);
                startPreprocess(full_tree{i}, sub_trees{i}, mode, '', dx_data);
            else
                parts = checkWhatMissing(sub_trees{i});
                printStatus(tree{i}, st.pet, st.fdg, st.mri);
                if(isempty(strfind([parts{:}], 'MRI')) == 1 && isempty(strfind([parts{:}], 'PET')) == 1)
                    startPreprocess(full_tree{i}, sub_trees{i}, mode, parts, dx_data);
                end
            end
            
            
        end
        
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
        
        function printStatus(patient, pet, fdg, mri)
            fprintf('Patient %s(%s) ...... ', patient, mode)
            fprintf('PET - ');
            cprintf('*comment', pet);
            fprintf(', FDG - ');
            cprintf('*comment', fdg);
            fprintf(', MRI - ');
            cprintf('*comment', mri);
            fprintf('\n');
        end
    end

    function startPreprocess(path, subpath, mode, missing, dx_data)
        fprintf('Preprocessing images: ......... ');
        f_locs = mfullfile(path, subpath);
               
        if(isempty(strfind(missing, 'FDG')) == 0) 
            mri = f_locs{2};
            fdg = cell(1,1); 
        else
            mri = f_locs{3};
            fdg = f_locs{2};
        end
        
        f = struct('pet', f_locs{1},'fdg', fdg, 'mri', mri);
        preprocessImages(f, mode, dx_data);
        fprintf('OK\n');
    end

end