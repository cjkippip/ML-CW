%{
Stock
k step ahead prediction
%}
clear;
clc;
data=importdata('report\assignment\FTSE100\20151127.csv');
dataNum=data.data;
dataC=dataNum(:,4);
dataVol=dataNum(:,5);
%%
dataC=flipud(dataC);
dataSize=size(dataC);
N=max(dataSize(1),dataSize(2));
Nts=20;
Ntr=N-Nts;
p=20;
Yts=dataC(Ntr+1:N,:);%test targets
%%
K=Nts;
predTs=ones(K,1);
for k=1:K
    Xtr=ones(Ntr-p+2-k,p);
    for i=1:Ntr-p+2-k
        for j=1:p-1
            Xtr(i,j)=dataC(i+j-1); %train data(matrix)      
        end
    end
    Ytr=dataC(p+k-1:Ntr,:);%train targets
 
    %neural network train
    X=Xtr';
    T=Ytr';
    [net]=feedforwardnet(20);
    [net]=train(net,X,T);
    [predTr]=net(X);
    predTr=predTr';

    xts=[dataC(Ntr-p+3-k:Ntr-k+1,:);1];
    predTs(k)=net(xts);
end
%%
x=1:Nts;
figure(9),clf,
plot(x,Yts,'r-','LineWidth',2);
hold on
plot(x,predTs,'b-','LineWidth',1);
legend('true', 'predicted value');
title('Compare on test data', 'FontSize', 14);
xlabel('Time', 'FontSize', 14);
ylabel('Value', 'FontSize', 14);
grid on
hold off
predictedClosePriceInNext10days=predTs;
predictedClosePriceInNext10days



