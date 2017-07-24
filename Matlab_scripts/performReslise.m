function files = performReslise(image1, realignTo)
    global spm_defaults;
    
    uData = {char(image1); char(realignTo)};
    % Adjust default parameters to use nearest neighbor interpolation
    param = spm_defaults.realign.write;
    param.interp = 0;
    job = struct('data', {uData}, 'roptions', param);
    files = spm_run_realign(job);    
end