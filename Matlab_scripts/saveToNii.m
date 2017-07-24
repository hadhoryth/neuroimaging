function path = saveToNii(header ,image, label)
[a, b, ext] = fileparts(header.fname);
path = strcat(label, b, ext);
path = fullfile(a, path);
header.fname = path;
%         header.private.dat.fname = header.fname;
spm_write_vol(header,image);
end