function Main_script(mode, needRearrange, needCSVRearrange)
global home spm_defaults config_defaults

% mode - 1 -> local settings
% mode - 2 -> sipba machine

load('spm_defaults.mat');
spm_defaults =  defaults;

if (mode == 1)
    home = '/Users/XT/Documents/PhD/Granada/neuroimaging';
    dataset_dir = mfullfile(home, 'dataset/');
    config_defaults = struct('cerebellum', {mfullfile(home, 'Atlas', 'Cerebellum-MNIsegment.nii')},...
                         'atlas', {mfullfile(home, 'Atlas', 'atlas116.nii')});
elseif (mode == 2)
    home = '/home/ivans';
    dataset_dir = '/home/ivans/Alzheimer_data/';
    config_defaults = struct('cerebellum', {'/home/ivans/Alzheimer_data/Cerebellum/Cerebellum-MNIsegment.nii'},...
                         'atlas', {'/home/ivans/Alzheimer_data/Atlas/atlas116.nii'});
end

if(exist('labels_from_csv.mat', 'file') ~= 2)
    xls_home = mfullfile(home,'csv_dataset/');
    xls_files = strcat(xls_home, {'ADNIMERGE.xlsx', 'DXSUM_PDXCONV_ADNIALL.xlsx'});
    mat_files = strcat(xls_home, {'data_ids.mat', 'data_bl_change.mat'});
    dx_labels = csv_rearrange(xls_home, xls_files, mat_files);
    save('labels_from_csv.mat', 'dx_labels');
else
    load('labels_from_csv.mat');
end

if (needRearrange == 1)
    info = rearangeFolders(dataset_dir, home, dx_labels);
    if(mode == 1)
        save ('info.mat', 'info');
    elseif (mode == 2)
        save ('info_sibpa.mat', 'info');
    end
else
    if(mode == 1)
        load('info.mat');
    elseif (mode == 2)
        load('info_sipba.mat');
    end
end

if(needCSVRearrange == 1)
    rearrangePreFolders(info, dx_labels);
end

[pathstr, name, ext] = fileparts(config_defaults.atlas);
pr_name = fullfile(pathstr, strcat('r', name, ext));
if(exist(pr_name, 'file') == 2)
    config_defaults.r_atlas = pr_name;
end

if(mode == 2)
    spm_path = mfullfile(home,'matlab_scripts','spm12');
    addpath (genpath(fullfile(home,'matlab_scripts','spm12')));
    spm_defaults.normalise.estimate.tpm = fullfile(spm_path, 'tpm','TPM.nii');
    
    tmp = {spm_defaults.segment.tissue.tpm};
    ngaus = {spm_defaults.segment.tissue.ngaus};
    native =  {spm_defaults.segment.tissue.native};
    warped = {spm_defaults.segment.tissue.warped};
    for i = 1 : length(tmp)
        [~, b, c] = fileparts(tmp{i}{1});
        tmp{i} = {mfullfile(spm_path, 'tpm', [b c])};
    end
    spm_defaults.segment.tissue = struct('tpm',tmp, 'ngaus',ngaus, 'native', native, 'warped',warped);
end


scanAndLogMissing(info, dx_labels);

end
