function files = preprocessImages(info, outFolder, dx_data)
global home config_defaults

o_name = 'ADNI_mat';
outputDir = mfullfile(home,o_name, outFolder);
if(exist(outputDir, 'dir') == 0)
    mkdir(outputDir);
end

outputDir_PET = mfullfile(outputDir, 'AV45');
if(exist(outputDir_PET, 'dir') == 0)
    mkdir(outputDir_PET);
end

outputDir_FDG = mfullfile(outputDir, 'FDG');
if(exist(outputDir_FDG, 'dir') == 0)
    mkdir(outputDir_FDG);
end

template = strcat(home, '/', 'templates/T1.nii');

totalLength = length(fieldnames(info));
toAnalize = struct('pet',[], 'fdg',[], 'mri',[]);
% for i = 1 : totalLength
    petDate = getDate(info.pet);
    if (isempty(info.fdg) == 0)
        fdgDate = getDate(info.fdg);
    end
    mriDate = getDate(info.mri);
    
    for j = 1 :length(petDate.year)
        pet_fdg = [];
        if (isempty(info.fdg) == 0)
            pet_fdg = pickValidDate(petDate.year(j), petDate.month(j), fdgDate, 1);
            if(length(pet_fdg) > 1)
                [~, fdg_i] = selectValidImage(fdgDate.image(pet_fdg));
                pet_fdg = pet_fdg(fdg_i);
            end
        end
        pet_mri = pickValidDate(petDate.year(j), petDate.month(j), mriDate, 1);    
                        
        if(length(pet_mri) > 1)
            [~, pet_i] = selectValidImage(mriDate.image(pet_mri));
            if(pet_i == 0), continue; end
            pet_mri = pet_mri(pet_i);
        elseif (isempty(pet_mri) == 1)
            continue;
        end
        
        if (length(mriDate.image{pet_mri}) > 1)            
            mriDate.image{pet_mri} = selectValidImage({mriDate.image{pet_mri}});
            if (isempty(mriDate.image{pet_mri}) == 1), continue; end
        end
        
        % set origin to PET and FDG image
        label = 'NewOrigin_';
        
        if(isFileExist(petDate.image{j}{1}, label) == 1)
            fprintf('FDG origin:');
            spmSetOrigin(petDate.image{j}{1}, template); 
            filename = getNewName(petDate.image{j}{1}, label);
            movefile(petDate.image{j}{1}, filename);
            petDate.image{j}{1} = filename;
            fprintf('OK; ');
        end
        
        
        if(isempty(pet_fdg) == 0)            
            if(isFileExist(fdgDate.image{pet_fdg}{1}, label) == 1)  
                fprintf('FDG origin: ');
                spmSetOrigin(fdgDate.image{pet_fdg}{1}, template);
                filename = getNewName(fdgDate.image{pet_fdg}{1}, label);
                movefile(fdgDate.image{pet_fdg}{1}, filename);
                fdgDate.image{pet_fdg}{1} = filename;
                fprintf('OK; ');
            end            
        end
              
        
       % preprocess PET based on MRI reference image  
       currentFDG = cell(1,1);
       if(isempty(pet_fdg) == 0)    
          currentFDG = fdgDate.image{pet_fdg};
       end
       [files_coreg, r] = isPreprocessed(petDate.image{j}{1}, currentFDG{1}, 'r', 'rfiles'); 
       if(r == 1)
%           try
             files_coreg = performCoregisterER( mriDate.image{pet_mri}, petDate.image{j}, currentFDG); 
%           catch Exception
%               if(strcmp(Exception.message, 'File too small.') == 1)
%                   fprintf('File corrupted. Trying another one');
%                   pet_mri = pickValidDate(petDate.year(j), petDate.month(j), mriDate, 2);                  
%                   files_coreg = performCoregisterER( mriDate.image{pet_mri}, petDate.image{j}, currentFDG);                    
%               end
%           end
       end
              
       [wr_files_norm, r] = isPreprocessed(petDate.image{j}{1}, currentFDG{1}, 'wr', 'wfiles');
       if(r == 1)
          wr_files_norm = performNormalizationEW (mriDate.image{pet_mri}, files_coreg.rfiles);
       end 
       
       % preprocess MRI images
       [w_files_norm, r] = isPreprocessed(mriDate.image{pet_mri}{1}, mriDate.image{pet_mri}{1}, 'w', 'wfiles'); 
       if(r == 1)
%           try
             w_files_norm = performNormalizationEW(mriDate.image{pet_mri}, mriDate.image{pet_mri});
%           catch Exception
%               if(strcmp(Exception.message, 'File too small.') == 1)
%                   fprintf('File corrupted. Trying another one');
%                   pet_mri = pickValidDate(petDate.year(j), petDate.month(j), mriDate, 2);
%                   w_files_norm = performNormalizationEW(mriDate.image{pet_mri}, mriDate.image{pet_mri});
%               end
%           end
       end
       
       [files_seg, r] = isPreprocessed(mriDate.image{pet_mri}{1}, mriDate.image{pet_mri}{1}, 'c1w', 'wfiles'); 
       if(r == 1)
          files_seg = performSegmentation(w_files_norm.wfiles);
       end
       
       % Cerebellum processing
       % Pet image has to be divided by SURv  
       [nCerImage, r] =  isPreprocessed(wr_files_norm.wfiles{1} , '', 'SURV_', 'files');
       if(r == 1)       
           nCerImage = normSUVR(config_defaults.cerebellum, wr_files_norm.wfiles{1}, 0);
       else
           nCerImage = char(nCerImage.files);
       end
       % FDG image normilized by 5% of the max intensity
%        if (isempty(wr_files_norm.wfiles{2}) == 0) 
       
       % Atlas processing       
       if(isfield(config_defaults, 'r_atlas') == 0)
          config_defaults.r_atlas = performReslise(w_files_norm.wrfiles{1}, config_defaults.atlas);
       end
       
       % Extraction mean intensities from SURv image 
       brainRegions_pet = getRegionFromAtlas(config_defaults.r_atlas, nCerImage);
       saveFeatures(nCerImage, 'brainRegions_pet', outputDir_PET, 'dx_data');
       
       if(length(wr_files_norm.wfiles) > 1)
           if(isempty(wr_files_norm.wfiles{2}) == 0)
               nFDGImage = normFDG(wr_files_norm.wfiles{2}, 0.05);
               brainRegions_fdg = getRegionFromAtlas(config_defaults.r_atlas, nFDGImage);
               saveFeatures(nFDGImage, 'brainRegions_fdg', outputDir_FDG, 'dx_data');
           end
       end
           
    end       
      
% end

    function [o_struct, isExists] = isPreprocessed(im1, im2, label, s_label)
        % isExist: 0 - it does; 1 - otherwise;
        isExists = 0;
        name1 = getNewName(im1, label);
        if(isempty(im2) == 1)
             o_struct = struct(s_label, {{name1}});
             if(exist(name1, 'file') == 0)                
                isExists = 1;
             end
             return;
        end
        name2 = getNewName(im2, label);
        o_struct = struct(s_label, {{name1, name2}});
        if(exist(name1, 'file') == 0 || exist(name2, 'file') == 0)
            isExists = 1;
        end
    end

    function name = getNewName(origin, add)
        [a, b, ext] = fileparts(origin);
        b = strcat(add, b, ext);
        name = mfullfile(a, b);
    end

    function date = getDate(dir)
        date = struct('year',[], 'month', [], 'image', []);
        subtree = checkHiddenFolders({dir});        
        for ii = 1 : length(subtree{1})
            current = subtree{1}{ii};
            if (regexp(current,'^\d\d\d\d-\d\d-\d\d') == 1)
                date.year(ii) = str2double(current(1:4));
                date.month(ii) = str2double(current(6:7));
%                 date.day(ii) = str2double(current(9:10));
                date.image{ii} = getImage();
            end
        end
        
        function path = getImage()
            full_path = mfullfile(dir, current);
            n_full_path = checkHiddenFolders({full_path});
            if(length(n_full_path{1}) > 1)           
                n_full_path = {getOriginalImages(n_full_path{1})};
            end
            full_path = mfullfile(full_path, n_full_path{1});
            if(isempty(strfind(full_path{1}, '.nii')) == 0)
                path = full_path;
            end
            
            function images = getOriginalImages(imgs)                
                k = 1;
                for iii = 1 : length(imgs)
                   if (startsWith(imgs{iii}, 'NewOrigin') == 1 || ...
                       startsWith(imgs{iii}, 'ADNI'))
                       images{k} = imgs{iii};
                       k = k + 1;
                   end
                end
                
                function b = startsWith(s, pat)
                    sl = length(s);
                    pl = length(pat);
                    b = (sl >= pl && strcmp(s(1:pl), pat)) || isempty(pat);
                end
            end
        end
        
    end
       
    function saveFeatures(im_dir, var, out_dir, var1)
        [~, b] = fileparts(im_dir);        
        save(mfullfile(out_dir, strcat(b, '.mat')),var, var1);        
    end

    function isExist = isFileExist(path, label)
        % isExist = 0 - exist
        %           1 - does not 
        isExist = 1;
        [a, b] = fileparts(path);
        if(isempty(strfind(b, label)) == 0)
            isExist = 0;
        end
        
    end

    function [image, index] = selectValidImage(images)
        images = [images{:}];
        index = 0;
        image = '';
        for kkk = 1 : length(images)
           current = char(images{kkk});
           header = spm_vol(current);
           try
               spm_read_vols(header);
               image = images(kkk);
               index = kkk;
               return;
           catch               
               [a,~] = fileparts(current);
               a = strsplit(a, '/');
               a = mfullfile(char(a(end-2)),char(a(end)));
               fprintf('\n%s - Corrupted, selecting next\n', a)
           end           
        end
    end

end