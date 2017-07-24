function spmSetOrigin(vols, template)
vols = vol1OnlySub(vols); %only process first volume of 4D datasets...
[pth,nam,ext, ~] = spm_fileparts(deblank(vols(1,:))); %extract filename
fname = fullfile(pth,[nam ext]); %strip volume label
%report if filename does not exist...
if (exist(fname, 'file') ~= 2)
    fprintf('%s set origin error: unable to find image %s.\n',mfilename,fname);
    return;
end;

coivox = ones(4,1);
hdr = spm_vol([fname,',1']);
img = spm_read_vols(hdr); %load image data
img = img - min(img(:));
img(isnan(img)) = 0;
sumTotal = sum(img(:));
coivox(1) = sum(sum(sum(img,3),2)'.*(1:size(img,1)))/sumTotal; %dimension 1
coivox(2) = sum(sum(sum(img,3),1).*(1:size(img,2)))/sumTotal; %dimension 2
coivox(3) = sum(squeeze(sum(sum(img,2),1))'.*(1:size(img,3)))/sumTotal; %dimension 3
XYZ_mm = hdr.mat * coivox; %convert from voxels to millimeters

for v = 1:   size(vols,1)
    fname = deblank(vols(v,:));
    if ~isempty(fname)
        [pth,nam,ext, ~] = spm_fileparts(fname);
        fname = fullfile(pth,[nam ext]);
        hdr = spm_vol([fname ',1']); %load header of first volume
        fname = fullfile(pth,[nam '.mat']);
        if exist(fname,'file')
            destname = fullfile(pth,[nam '_old.mat']);
            copyfile(fname,destname);
            fprintf('%s is renaming %s to %s\n',mfilename,fname,destname);
        end
        hdr.mat(1,4) =  hdr.mat(1,4) - XYZ_mm(1);
        hdr.mat(2,4) =  hdr.mat(2,4) - XYZ_mm(2);
        hdr.mat(3,4) =  hdr.mat(3,4) - XYZ_mm(3);
        spm_create_vol(hdr);
        if exist(fname,'file')
            delete(fname);
        end
    end
end
coregSub(vols);
for v = 1:   size(vols,1)
    [pth, nam, ~, ~] = spm_fileparts(deblank(vols(v,:)));
    fname = fullfile(pth,[nam '.mat']);
    if exist(fname,'file')
        delete(fname);
    end
end

    function coregSub(vols)
        matlabbatch{1}.spm.spatial.coreg.estimate.ref = {template};
        matlabbatch{1}.spm.spatial.coreg.estimate.source = {[deblank(vols(1,:)),',1']};
        matlabbatch{1}.spm.spatial.coreg.estimate.other = cellstr(vols(2:end,:));% {''};
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
        matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
        spm_jobman('run',matlabbatch);
    end

end

function vols = vol1OnlySub(vols)
oldvols = vols;
vols = [];
    for v = 1:   size(oldvols,1)
        [pth,nam,ext,n] = spm_fileparts(deblank(oldvols(v,:)));
        if n > 1
            error('%s error: for 4D images please only specify the first volume of the series',mfilename);
        end
        vols = strvcat(vols, fullfile(pth, [ nam ext ',1']) );
    end
end