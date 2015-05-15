% Train AdaBoost 
% http://de.mathworks.com/help/stats/fitensemble.html


%% read training images
fid = fopen('/Users/nanliu/Documents/MATLAB/MNIST/train-images.idx3-ubyte','r'); 
a = fread(fid,16,'uint8'); 
% get information
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12);
ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);
% if is train-images.idx3-ubyte file
if ((MagicNum~=2051)||(ImageNum~=60000))
    error('Not MNIST train-images.idx3-ubyte!');
    fclose(fid);    
    return;    
end 
% read data
TrainImages = zeros(ImageNum,ImageRow*ImageCol);
for i=1:ImageNum
    TrainImages(i,:) = fread(fid,ImageRow*ImageCol,'uint8');   
end
fclose(fid);

%% read test images
fid = fopen('/Users/nanliu/Documents/MATLAB/MNIST/t10k-images.idx3-ubyte','r'); 
a = fread(fid,16,'uint8'); 
% get information
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ImageNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
ImageRow = ((a(9)*256+a(10))*256+a(11))*256+a(12);
ImageCol = ((a(13)*256+a(14))*256+a(15))*256+a(16);
% if is t10k-images.idx3-ubyte file
if ((MagicNum~=2051)||(ImageNum~=10000))
    error('Not MNIST t10k-images.idx3-ubyte!');
    fclose(fid);    
    return;    
end 
% read data
TestImages = zeros(ImageNum,ImageRow*ImageCol);
for i=1:ImageNum
    TestImages(i,:) = fread(fid,ImageRow*ImageCol,'uint8');   
end
fclose(fid);



%% read training image labels
fid = fopen('/Users/nanliu/Documents/MATLAB/MNIST/train-labels.idx1-ubyte','r'); 
a = fread(fid,8,'uint8'); 
% get info
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ItemNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
% if is train-labels.idx1-ubyte file
if ((MagicNum~=2049)||(ItemNum~=60000))
    error('Not MNIST train-labels.idx1-ubyte');
    fclose(fid);    
    return;    
end 
% read labels
TrainLabels = fread(fid,ItemNum,'uint8');   
fclose(fid);

%% read test image labels
fid = fopen('/Users/nanliu/Documents/MATLAB/MNIST/t10k-labels.idx1-ubyte','r'); 
a = fread(fid,8,'uint8'); 
% get info
MagicNum = ((a(1)*256+a(2))*256+a(3))*256+a(4);
ItemNum = ((a(5)*256+a(6))*256+a(7))*256+a(8);
% if is t10k-labels.idx1-ubyte file
if ((MagicNum~=2049)||(ItemNum~=10000))
    error('Not MNIST t10k-labels.idx1-ubyte');
    fclose(fid);    
    return;    
end 
% read labels
TestLabels = fread(fid,ItemNum,'uint8');   
fclose(fid);


%% train classification tree ensemble using AdaBoost
%ClassTreeEns = fitensemble(TrainImages(1:1000,:),TrainLabels(1:1000),'AdaBoostM2',1000,'Tree');
ClassTreeEns = fitensemble(TrainImages(1:1000,:),TrainLabels(1:1000),'LPBoost',1000,'Tree');

%% check against test set
L = loss(ClassTreeEns,TestImages,TestLabels);
disp('the classification error is :'); 
disp(L);

%% play with the number of learning rounds and visualize the 
rsLoss = resubLoss(ClassTreeEns,'Mode','Cumulative');
plot(rsLoss);
xlabel('Number of Learning Cycles');
ylabel('Resubstitution Loss');