%{
Chaotic Time Series
one step ahead prediction
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
Xts=ones(Nts,p);
for i=1:Nts
    for j=1:p-1
        Xts(i,j)=timeSerie(1481+i+j-1); %test data(matrix)      
    end
end
Ytr=timeSerie(p:Ntr,:);%train targets
Yts=timeSerie(Ntr+1:N,:);%test targets
%%
%linear train
w=(Xtr'*Xtr) \ Xtr'*Ytr;
% w=Xtr \ Ytr;
predTr1=Xtr*w;

predTs1=Xts*w;
errTr1=norm(predTr1-Ytr)^2/(Ntr-p+1);
errTs1=norm(predTs1-Yts)^2/Nts;

x=1:Nts;
figure(4),clf,
plot(x,Yts,'r-','LineWidth',4);
hold on
plot(x,predTs1,'b--','LineWidth',2);
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

predTs2=net(Xts');
errTr2=norm(predTr2'-Ytr)^2/(Ntr-p+1);
errTs2=norm(predTs2'-Yts)^2/Nts;

figure(5),clf,
plot(x,Yts,'r-','LineWidth',4);
hold on
plot(x,predTs2,'b--','LineWidth',2);
legend('true', 'predicted value');
title('Compare on test data', 'FontSize', 14);
xlabel('Time', 'FontSize', 14);
ylabel('Value', 'FontSize', 14);
grid on
hold off


