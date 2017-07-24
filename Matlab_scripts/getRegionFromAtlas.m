function regions = getRegionFromAtlas(atlasPath, imagesPath)
    H_A = spm_vol(char(atlasPath));
    V_A = spm_read_vols(H_A);
    V_A(find(isnan(V_A))) = 0;
    nROIs = max(V_A(:));
    
    meanIntensities = [];
    
    for i = 1:size(imagesPath,1)
        headerInfo = spm_vol(imagesPath);
        vox = spm_read_vols(headerInfo);
        vox(find(isnan(vox))) = 0;
        
        for roi = 1:nROIs
            voxRoi = vox(find(V_A == roi));
            meanIntensities = [meanIntensities, mean(voxRoi)];
        end
        
    end
    
    regions = meanIntensities;
end