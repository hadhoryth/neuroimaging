function files = performSegmentation(toSegment)

global spm_defaults
% MRI defaults is using, defining hard-coding
ch = struct('vols', {toSegment}, 'biasreg',1e-3, ... 
                 'biasfwhm', 60, 'write', [0 0]);
job = struct('channel', ch, 'tissue', spm_defaults.segment.tissue, ...
             'warp', spm_defaults.segment.warp);
         
files = spm_preproc_run(job, 'run');
end