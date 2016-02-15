%{
Stock
k step ahead prediction
NO TEST DATA
%}
clear;
clc;
data=importdata('report\assignment\FTSE100\20151209.csv');
dataNum=data.data;
dataC=dataNum(:,4);
dataVol=dataNum(:,5);
%%
dataC=flipud(dataC);
dataSize=size(dataC);
Ntr=max(dataSize(1),dataSize(2));
p=20;
Xtr=ones(Ntr-p+1,p);
for i=1:Ntr-p+1
    for j=1:p-1
        Xtr(i,j)=dataC(i+j-1); %train data(matrix)      
    end
end
Ytr=dataC(p:Ntr,:);%train targets
%%
%neural network train
X=Xtr';
T=Ytr';
[net]=feedforwardnet(20);
[net]=train(net,X,T);
[predTr]=net(X);

xts=[dataC(Ntr-p+2:Ntr,:);1];
predTs=net(xts);
predictedClosePriceInNext10days






