close all; clear all; clc;


global home spm_defaults config_defaults
% home = '/Users/XT/Documents/MATLAB';
% dataset_dir = '/Users/XT/Documents/PhD/Granada/dataset/';

home = '/home/ivans';
dataset_dir = '/home/ivans/Alzheimer_data/';

load('spm_defaults.mat');
spm_defaults =  defaults;
% config_defaults = struct('cerebellum', {'/Users/XT/Documents/PhD/Granada/MatLab/Test/Cerebellum/Cerebellum-MNIsegment.nii'},...
%                          'atlas', {'/Users/XT/Documents/PhD/Granada/MatLab/Test/Atlas/atlas116.nii'});

config_defaults = struct('cerebellum', {'/home/ivans/Alzheimer_data/Cerebellum/Cerebellum-MNIsegment.nii'},...
                         'atlas', {'/home/ivans/Alzheimer_data/Atlas/atlas116.nii'});



info = rearangeFolders(dataset_dir, home);
save ('info.mat');

%load('info.mat');
[pathstr, name, ext] = fileparts(config_defaults.atlas);
pr_name = fullfile(pathstr, strcat('r', name, ext));
if(exist(pr_name, 'file') == 2)
    config_defaults.r_atlas = pr_name;
end

spm_path = mfullfile(home,'matlab_scripts','spm12');
addpath (genpath(spm_path));
spm_defaults.normalise.estimate.tpm = {fullfile(spm_path, 'tpm','TPM.nii')};

%Update spm_defaults 
tmp = {spm_defaults.segment.tissue.tpm};
ngaus = {spm_defaults.segment.tissue.ngaus};
native =  {spm_defaults.segment.tissue.native};
warped = {spm_defaults.segment.tissue.warped};
for i = 1 : length(tmp)
    [a, b, c] = fileparts(tmp{i}{1});
    tmp{i} = {mfullfile(spm_path, 'tpm', [b c])};
end
spm_defaults.segment.tissue = struct('tpm',tmp, 'ngaus',ngaus, 'native', native, 'warped',warped); 


scanAndLogMissing(info);
