% Matlab Read t10k-labels_idx1 and train-labels.idx1


[FileName,PathName] = uigetfile('*.*','train-labels.idx1-ubyte');
TrainFile = fullfile(PathName,FileName);
fid = fopen(TrainFile,'r'); 
a = fread(fid,8,'uint8'); 

%% get info
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ItemNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);

%% if is t10k-labels.idx1-ubyte file
if ((MagicNum~=2049)||(ItemNum~=60000))
    error('Not MNIST train-labels.idx1-ubyte');
    fclose(fid);    
    return;    
end 

%% read labels
savedirectory = uigetdir('','train-labels');
TrainLabels = fread(fid,ItemNum,'uint8');   
save TrainLabels;
fclose(fid);

