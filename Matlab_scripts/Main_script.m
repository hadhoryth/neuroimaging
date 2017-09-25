clear all; close all; clc;
global home home_out spm_defaults config_defaults cleanData data_ids

running_mode = 1;
cleanData = 1;

load('spm_defaults.mat');
spm_defaults =  defaults;

if (running_mode == 1)
    home = '/Users/XT/Documents/PhD/Granada/neuroimaging';
    % External HDD
    dataset_dir = mfullfile('/Volumes/ELEMENT/Alzheimer');
    % dataset_dir = mfullfile(home, 'dataset/');
    config_defaults = struct('cerebellum', {mfullfile(home, 'Atlas', 'Cerebellum-MNIsegment.nii')},...
        'atlas', {mfullfile(home, 'Atlas', 'atlas116.nii')});
elseif (running_mode == 2)
    home = '/home/ivans';
    dataset_dir = '/home/ivans/Alzheimer_data/';
    config_defaults = struct('cerebellum', {'/home/ivans/Alzheimer_data/Cerebellum/Cerebellum-MNIsegment.nii'},...
        'atlas', {'/home/ivans/Alzheimer_data/Atlas/atlas116.nii'});
end

xls_home = mfullfile(home,'csv_dataset/');
config_defaults.xls_files = strcat(xls_home, {'ADNIMERGE.xlsx', 'DXSUM_PDXCONV_ADNIALL.xlsx'});
config_defaults.mat_files = strcat(xls_home, {'data_ids.mat', 'data_bl_change.mat'});
if(exist('labels_from_csv.mat', 'file') ~= 2 && exist(config_defaults.mat_files{1}, 'file') ~= 2)    
    dx_labels = csv_rearrange(xls_home, config_defaults.xls_files, config_defaults.mat_files);
    save('labels_from_csv.mat', 'dx_labels');
else
    load('labels_from_csv.mat');
    tmp = load(config_defaults.mat_files{1}); 
    data_ids = tmp.data_ids;
end

[pathstr, name, ext] = fileparts(config_defaults.atlas);
pr_name = fullfile(pathstr, strcat('r', name, ext));
if(exist(pr_name, 'file') == 2)
    config_defaults.r_atlas = pr_name;
end

spm_path = mfullfile(home,'Matlab_scripts','spm12');
addpath (genpath(fullfile(home,'Matlab_scripts','spm12')));
spm_defaults.normalise.estimate.tpm = {fullfile(spm_path, 'tpm','TPM.nii')};

tmp = {spm_defaults.segment.tissue.tpm};
ngaus = {spm_defaults.segment.tissue.ngaus};
native =  {spm_defaults.segment.tissue.native};
warped = {spm_defaults.segment.tissue.warped};
for i = 1 : length(tmp)
    [~, b, c] = fileparts(tmp{i}{1});
    tmp{i} = {mfullfile(spm_path, 'tpm', [b c])};
end
spm_defaults.segment.tissue = struct('tpm',tmp, 'ngaus',ngaus, 'native', native, 'warped',warped);


home_out = '/Volumes/ELEMENT/Alzheimer';
scanAndLogMissing('/Volumes/ELEMENT/Alzheimer/ADNI_Rearranged', dx_labels);
% rearrangePreFolders('/Volumes/ELEMENT/Alzheimer/ADNI_Rearranged', dx_labels);
