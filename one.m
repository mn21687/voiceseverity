ADS=audioDatastore('D:\prediction\PATHOL\*-0-*.wav');    %% read all file of 0 severity 
ADS.Files{1}    %%  index 1 file 
[y,fs]=audioread(ADS.Files{1});   %%  reads data from the file named filename, and returns sampled data, y,  and a sample rate for that data, Fs. 
sound(y,fs)       %%  listen a sound 
coeffs = mfcc(y,fs)        %%  Compute Mel Frequency Cepstral Coefficients
%% we have alreadycompute Compute Mel Frequency Cepstral Coefficients of 1st file 
%% now we create a loop and append other file mfcc below coeffs variable 
numberOfRow = size(ADS.Files,1);  %% In ADS variable we have 7 files
for i=2:numberOfRow  %% start loop that make coeffs 2 to 7 file 
[y,fs]=audioread(ADS.Files{i})
Temp = mfcc(y,fs)
coeffs = [coeffs;Temp]
end;

ADS=audioDatastore('D:\prediction\NORM\*-0-*.wav');    %% read all file of 0 severity from Norm folder 
ADS.Files{1}    %%  index 1 file 
[y,fs]=audioread(ADS.Files{1});   %%  reads data from the file named filename, and returns sampled data, y,  and a sample rate for that data, Fs. 
sound(y,fs)       %%  listen a sound 
coeffs0 = mfcc(y,fs)        %%  Compute Mel Frequency Cepstral Coefficients
%% we have alreadycompute Compute Mel Frequency Cepstral Coefficients of 1st file 
%% now we create a loop and append other file mfcc below coeffs variable 
numberOfRow = size(ADS.Files,1);  %% In ADS variable we have 119 files
for i=2:numberOfRow  %% start loop that make coeffs 2 to 119 file 
[y,fs]=audioread(ADS.Files{i})
Temp = mfcc(y,fs)
coeffs0 = [coeffs0;Temp]
end

coeffs=[coeffs;coeffs0];    %% in coeffs we have 41652 rows

ADS=audioDatastore('D:\prediction\PATHOL\*-1-*.wav');  %% read all file of 1 severity that contain 46 severity 
[y,fs]=audioread(ADS.Files{1});   %%  reads data from the file named filename, and returns sampled data, y,  and a sample rate for that data, Fs. 
coeff1 = mfcc(y,fs)     %%  Compute Mel Frequency Cepstral Coefficients
%% we have alreadycompute Compute Mel Frequency Cepstral Coefficients of 1st file 
%% now we create a loop and append other file mfcc below coeff1 variable
numberOfRow = size(ADS.Files,1);  %% In ADS variable we have 46 file 
for i=2:numberOfRow 
[y,fs]=audioread(ADS.Files{i})
Temp = mfcc(y,fs)
coeff1 = [coeff1;Temp]
end


ADS=audioDatastore('D:\prediction\PATHOL\*-2-*.wav'); %% read all file of 2 severity
[y,fs]=audioread(ADS.Files{1}); 
coeff2 = mfcc(y,fs)  %%  Compute Mel Frequency Cepstral Coefficients
%% we have alreadycompute Compute Mel Frequency Cepstral Coefficients of 1st file 
%% now we create a loop and append other file mfcc below coeff2 variable
numberOfRow = size(ADS.Files,1);  %% In ADS variable we have 37 file
for i=2:numberOfRow 
[y,fs]=audioread(ADS.Files{i})
Temp = mfcc(y,fs)
coeff2 = [coeff2;Temp]
end


ADS=audioDatastore('D:\prediction\PATHOL\*-3-*.wav');   %% read all file of 3 severity
[y,fs]=audioread(ADS.Files{1});
coeff3 = mfcc(y,fs);    %%  Compute Mel Frequency Cepstral Coefficients

%% we have alreadycompute Compute Mel Frequency Cepstral Coefficients of 1st file 
%% now we create a loop and append other file mfcc below coeff2 variable
numberOfRow = size(ADS.Files,1);  %% In ADS variable we have 15 file
for i=2:numberOfRow 
[y,fs]=audioread(ADS.Files{i})
Temp = mfcc(y,fs)
coeff3 = [coeff3;Temp]
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numberOfRow = size(coeffs,1);  %% In coeffs variable we have 41652 number Of Rows
%% For assign label 1 to 41652 rows  create a loop that make a column of 1's that contain 1734 rows 
A = zeros(numberOfRow,1);        
for k=1:numberOfRow
A(k,:) =  0 ; % just an example
end
%%Add 15 label coulmn in coeffs we use this funtion 
ind = [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ];
M2result = zeros(size(coeffs,1),numel(ind));
gind = logical(ind);
M2result(:,gind) = A;
M2result(:,~gind) = coeffs;


numberOfRow = size(coeff1,1);  %% In coeff1 variable we have 10775 number Of Rows
%% For assign label 0 to 10775  %% rows  create a loop that make a column of 0's that contain 10775 rows 
A = zeros(numberOfRow,1);
for k=1:numberOfRow  
A(k,:) =  1 ; % just an example
end
%%Add 15 label coulmn in coeff1 we use this funtion
ind = [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ];
M2result1 = zeros(size(coeff1,1),numel(ind));
gind = logical(ind);
M2result1(:,gind) = A;
M2result1(:,~gind) = coeff1;


numberOfRow = size(coeff2,1);  %% In coeff2 variable we have 9264 number Of Rows
%% For assign label 0 to 9264 rows  create a loop that make a column of 0's that contain 9264 rows
A = zeros(numberOfRow,1);
for k=1:numberOfRow 
A(k,:) =  0 ; %% Assign zero 
end
%%Add 15 label coulmn in coeff2 we use this funtion
ind = [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ];
M2result2 = zeros(size(coeff2,1),numel(ind));
gind = logical(ind);
M2result2(:,gind) = A;
M2result2(:,~gind) = coeff2;


numberOfRow = size(coeff3,1);  %% In coeff3 variable we have 3271 number Of Rows
A = zeros(numberOfRow,1);
for k=1:numberOfRow 
A(k,:) =  0 ; % just an example
end
%%Add 15 label coulmn in coeff3 we use this funtion
ind = [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 ];
M2result3 = zeros(size(coeff3,1),numel(ind));
gind = logical(ind);
M2result3(:,gind) = A;
M2result3(:,~gind) = coeff3;



M2result1=[M2result1;M2result;M2result2;M2result3]; 
M2result1 = M2result1(randperm(size(M2result1, 1)), :);
X=M2result1(1:64962,1:14);    %% make a X prediction 
Y=M2result1(1:64962,15:15);   %% make a Y label 


c = cvpartition(Y, 'KFold', 10); % create stratified folds
kSVMModel = fitcsvm(X,Y, 'Standardize', true, 'CVPartition', c);
scorekSVMModel = fitSVMPosterior(kSVMModel);
[predictions, post_scores] = kfoldPredict(scorekSVMModel);


for jj = 1:kSVMModel.KFold % debug
indTrainFold{jj} = find(training(c,jj)==1);
indTestFold{jj} = find(test(c,jj)==1);
[predFold{jj}] = predict(kSVMModel.Trained{jj}, X(indTestFold{jj},:));
cmFold = confusionchart(Y(indTestFold{jj},:), predFold{jj});
TN(jj) = cmFold.NormalizedValues(1,1);
TP(jj) = cmFold.NormalizedValues(2,2);
FP(jj) = cmFold.NormalizedValues(1,2);
FN(jj) = cmFold.NormalizedValues(2,1);
close all;
end
TN=sum(TN);
TP=sum(TP);
FP=sum(FP);
FN=sum(FN);
sensitivity= TP/(TP+FN)*100;     %% To find sensitivity
specificity=TN/(TN+FP)*100;       %% To find specificity
Accuracy=(TP+TN)/(TP+TN+FP+FN)*100; %% To find Accuracy

cm = confusionchart(Y, predictions);
cm.Title = 'For one severity';
sum(TN) == cm.NormalizedValues(1,1);
sum(TP) == cm.NormalizedValues(2,2);
sum(FP) == cm.NormalizedValues(1,2);
sum(FN) == cm.NormalizedValues(2,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Zero serverity
x_train = X(1:51970,:)
x_test  = X(51970:end,:);
y_train = Y(1:51970,:);
y_test  = Y(51970:end,:)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Mdl = fitcknn(x_train,y_train,'NumNeighbors',5,'ClassNames',{'1','0'},'Distance','euclidean', 'Standardize',1);
pred = str2num(cell2mat(predict(Mdl,x_test)));
cm = confusionchart(y_test, pred);
cm.Title = 'For zero severity';
accuracy=confusionmatStats(y_test,pred);
disp('KNN classifier results for one severity');
aTable = struct2table(accuracy); disp(aTable);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Dtree for zero
tree = fitctree(x_train,y_train);
predictLabels = predict(tree,x_test); 
testAccuracy = sum(predictLabels == y_test)/length(y_test);
cm = confusionchart(y_test,predictLabels);
cm.Title = 'For one severity';
accuracy=confusionmatStats(y_test,predictLabels);
disp('Decision tree classifier results for one severity');
aTable = struct2table(accuracy); disp(aTable);
