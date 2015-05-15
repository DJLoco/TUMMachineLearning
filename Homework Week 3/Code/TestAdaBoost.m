% Test AdaBoost

%% train classification tree ensemble using AdaBoost
ClassTreeEns = fitensemble(TrainImages,TrainLabels,'AdaBoostM2',750,'Tree');

%% check against test set
L = loss(ClassTreeEns,TestImages,TestLabels);
disp('the classification error is :'); 
disp(L);

