%{
Chaotic Time Series
free running mode
%}
clear;
clc;
N=2000;
timeSerie=mackeyglass(N);
timeSerie=timeSerie(2:N+1,:);
%%
Ntr=1500;
Nts=N-Ntr;
p=20;
Xtr=ones(Ntr-p+1,p);
for i=1:Ntr-p+1
    for j=1:p-1
        Xtr(i,j)=timeSerie(i+j-1); %train data(matrix)      
    end
end
Ytr=timeSerie(p:Ntr,:);%train targets
Yts=timeSerie(Ntr+1:N,:);%test targets
%%
%linear train
w=(Xtr'*Xtr) \ Xtr'*Ytr;
% w=Xtr \ Ytr;
predTr1=Xtr*w;

%one step ahead prediction (linear)
oneStepSerie1=[timeSerie(1482:1500,:);1];%initialize [1 ... 20]
predSerie1=ones(Nts,1);%ones predicted output vector
for ii=1:Nts
    predVal1=oneStepSerie1'*w;%predicted output value
    %replace and sort 
    for jj=1:p-2
        oneStepSerie1(jj)=oneStepSerie1(jj+1);
    end
    oneStepSerie1(p-1)=predVal1;
    
    predSerie1(ii)=predVal1;
end
errTr1=norm(predTr1-Ytr)^2/(Ntr-p+1);
errTs1=norm(predSerie1-Yts)^2/Nts;

x=1:Nts;
figure(6),clf,
plot(x,Yts,'r-','LineWidth',2);
hold on
plot(x,predSerie1,'b-','LineWidth',1);
legend('true', 'predicted value');
title('Compare on test data', 'FontSize', 14);
xlabel('Time', 'FontSize', 14);
ylabel('Value', 'FontSize', 14);
grid on
hold off

%%
%neural network train
X=Xtr';
T=Ytr';
[net]=feedforwardnet(20);
[net]=train(net,X,T);
[predTr2]=net(X);
predTr2=predTr2';

%one step ahead prediction (NN)
oneStepSerie2=[timeSerie(Ntr-p+2:Ntr,:);1];%initialize [1 ... 20]'
predSerie2=ones(Nts,1);%ones predicted output vector
for ii=1:Nts
    predVal2=net(oneStepSerie2);%predicted output value
    %replace and sort 
    for jj=1:p-2
        oneStepSerie2(jj)=oneStepSerie2(jj+1);
    end
    oneStepSerie2(p-1)=predVal2;
    
    predSerie2(ii)=predVal2;
end
errTr2=norm(predTr2-Ytr)^2/(Ntr-p+1);
errTs2=norm(predSerie2-Yts)^2/Nts;

figure(7),clf,
plot(x,Yts,'r-','LineWidth',2);
hold on
plot(x,predSerie2,'b-','LineWidth',1);
legend('true', 'predicted value');
title('Compare on test data', 'FontSize', 14);
xlabel('Time', 'FontSize', 14);
ylabel('Value', 'FontSize', 14);
grid on
hold off


