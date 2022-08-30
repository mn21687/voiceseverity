# voiceseverity
https://drive.google.com/drive/u/0/folders/1zUbpCKJ0zp8E_RwDTyCHjA78NL4ppwiN
That is the link of gmail drive where AVPD database store you can download pathol or normal voice. 
So you can update Path in code ADS=audioDatastore('D:\prediction\PATHOL\*-0-*.wav'); where directory you store AVPD database and run the code 
There are three classifier in each severity file SVM and KNN and Decision Tree 
SVM take more time for excution As compare KNN and Decision tree so Wait when SVm run and show you results . 
In each Severity you should update your path in Audiostore object 
For check the results Accuracy , Sensitivity , specificity , Percision , Recall , F1score classOrder you can Add function with name confusionmatStats. and use in code accuracy=confusionmatStats(y_test,predictLabels); like this
