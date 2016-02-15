function [predTr,predTs,errTr,errTs]=oneStepAheadNN(data,N,Ntr)
Nts=N-Ntr;
p=20;
Xtr=ones(Ntr-p+1,p);
for i=1:Ntr-p+1
    for j=1:p-1
        Xtr(i,j)=data(i+j-1); %train data(matrix)      
    end
    Xtr(i,p)=1;
end
Ytr=data(p:Ntr,:);%train targets
Yts=data(Ntr+1:N,:);%test targets
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
end

