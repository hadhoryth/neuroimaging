function nImagePath = normFDG(fdg_image, percent)
header = spm_vol(fdg_image);
im = spm_read_vols(header);
maxC = max(im(:)) * percent;
nImagePath = saveToNii(header, im./maxC, 'FDG_');
end