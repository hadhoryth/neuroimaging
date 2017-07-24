function nImagePath = normSUVR(suvImg, toNormilize, needShow)
[data1, data2, h1, h2] = readImagesVols(suvImg, toNormilize);
meanC = mean(data2(data1(:) > 0)) ;
nImage = data2./meanC;

%   showMontage(data1);
if needShow == 1
    showMontage(data2);
    title('Original image');
    
    showMontage(nImage);
    title('SUVr image');
end

nImagePath = saveToNii(h2, nImage, 'SURV_');

    function showMontage(image)
        %% normilize montage function cause in >> 1
        [x y z] = size(image);
        reshapedImg = reshape(image,[x y 1 z]);
        figure;
        montage(reshapedImg);
    end

end



function [im, im1, header1, header2] = readImagesVols(a, b)
header1 = spm_vol(a);
header2 = spm_vol(b);

im = spm_read_vols(header1);
im1 = spm_read_vols(header2);
end

