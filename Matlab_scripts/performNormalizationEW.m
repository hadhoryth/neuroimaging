function files = performNormalizationEW (toAlign, toWrite)
    % resample cell has to be column wise
    global spm_defaults;

    subject = struct('vol', {toAlign}, 'resample', {toWrite'});    
    job = struct('subj',subject,'eoptions', spm_defaults.normalise.estimate, 'woptions', spm_defaults.normalise.write);

    files = spm_run_norm(job); 
    files = struct('wfiles', {files.files}, 'def', {files.def});
end