%cleanup
clearvars
close all
myDir = uigetdir;
person = 'Actor5_ravdess';
final = {'Person' 'Filename' 'Length' 'IMF'};

myFiles = dir(fullfile(myDir,'*.wav')); %gets all wav files in struct
for k = 1:length(myFiles)
    baseFileName = myFiles(k).name;
    
    fullFileName = fullfile(myDir, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    [wave,fs]=audioread(fullFileName);
    t=0:1/fs:(length(wave)-1)/fs;
    param.n_grid_1    = 500;  %Grid size in dimension 1
    param.nimfs       = 5;   %Maximum number of IMFs that can be stored
    param.type        = 5; %type of window size
    param.tol         = 0.05; %sifting tolerance
    param.plot        = 'off'; %plots on
    results = EMD1DNV(wave,t ,param);
    a = {person baseFileName size(wave,1) results.IMF};
    final = cat(1,final,a);
end

writecell(final ,'Actor4_ravdess.csv')