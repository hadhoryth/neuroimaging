function files = performCoregisterER (refImagePath, sourceImagePath, otherImagesPath)

   global spm_defaults;

    job = struct('ref', {refImagePath},'source', {sourceImagePath}, 'other', {otherImagesPath},...
        'eoptions', spm_defaults.coreg.estimate,'roptions', spm_defaults.coreg.write);
    
   files = spm_run_coreg(job);   
    
end