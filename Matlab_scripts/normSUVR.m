function normilized_image_path = normSUVR(cerebellum, pet_image, white_matter,needShow)
cerebellum = performCoregisterER({white_matter}, {cerebellum}, cell(1,1));
cerebellum = cerebellum.rfiles{1}(1:end-2);

cer_vols = readImageVols(cerebellum);
[pet_vols, pet_header] = readImageVols(pet_image);
[wm_vols, wm_header] = readImageVols(white_matter);


mask = cer_vols;
mask(mask > 0) = 1;
wm_cerebellum = wm_vols .* mask;
normilized_pet = pet_vols./mean(wm_cerebellum(:));
% saveToNii(pet_header, normilized_wm, 'WM_')

% meanC = mean(pet_vols(cer_vols(:) > 0)) ;
% nImage = pet_vols./meanC;

%   showMontage(data1);
if needShow == 1
    showMontage(data2);
    title('Original image');
    
    showMontage(nImage);
    title('SUVr image');
end

normilized_image_path = saveToNii(pet_header, normilized_pet, 'SUVR_');

delete(cerebellum)

    function showMontage(image)
        %% normilize montage function cause in >> 1
        [x y z] = size(image);
        reshapedImg = reshape(image,[x y 1 z]);
        figure;
        montage(reshapedImg);
    end

end

function [vols, header] = readImageVols(image_path)
    header = spm_vol(image_path);
    vols = spm_read_vols(header);
end


