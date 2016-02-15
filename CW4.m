%{
Stock
free running mode
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
Xtr=ones(Ntr-p+1,p);
for i=1:Ntr-p+1
    for j=1:p-1
        Xtr(i,j)=dataC(i+j-1); %train data(matrix)      
    end
end
Ytr=dataC(p:Ntr,:);%train targets
Yts=dataC(Ntr+1:N,:);%test targets
%%
%neural network train
X=Xtr';
T=Ytr';
[net]=feedforwardnet(20);
[net]=train(net,X,T);
[predTr]=net(X);
predTr=predTr';

%one step ahead prediction (NN)
oneStepSerie=Xtr(Ntr-p+1,:)';%initialize [1 ... 20]'
predTs=ones(Nts,1);%ones predicted output vector
for ii=1:Nts
    predVal=net(oneStepSerie);%predicted output value
    %replace and sort 
    for jj=1:p-2
        oneStepSerie(jj)=oneStepSerie(jj+1);
    end
    oneStepSerie(p-1)=predVal;
    
    predTs(ii)=predVal;
end
errTr=norm(predTr-Ytr)^2/(Ntr-p+1);
errTs=norm(predTs-Yts)^2/Nts;
%%
x=1:Nts;
figure(8),clf,
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
% x=1:N;
% y1=dataC;
% y2=[dataC(1:19,:);predTr;predTs];
% figure(8),clf,
% plot(x,y1,'r-','LineWidth',2);
% hold on
% plot(x,y2,'b-','LineWidth',1);
% legend('true', 'predicted value');
% title('Compare on test data', 'FontSize', 14);
% xlabel('Time', 'FontSize', 14);
% ylabel('Value', 'FontSize', 14);
% grid on
% hold off

% data1=importdata('assignment/FTSE100/close.csv');
% data2=importdata('assignment/FTSE100/high.csv');
% data3=importdata('assignment/FTSE100/low.csv');
% data4=importdata('assignment/FTSE100/open.csv');
% [predSerie1,errTr1]=oneStepAhead(data1,5);
% [predSerie2,errTr2]=oneStepAhead(data2,5);
% [predSerie3,errTr3]=oneStepAhead(data3,5);
% [predSerie4,errTr4]=oneStepAhead(data4,5);
% predTable=[predSerie1 predSerie2 predSerie3 predSerie4];
















