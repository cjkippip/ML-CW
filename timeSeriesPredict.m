%{
Stock
k step ahead prediction
use matlab function
%}
data=importdata('report\assignment\FTSE100\20151127.csv');
dataNum=data.data;
dataC=dataNum(:,4);
dataVol=dataNum(:,5);

na = 1;
nb = 2;
sys = armax(dataC(1:200),[na nb]);

K = 4;
yp = predict(sys,dataC,K);
x=1:200;
plot(x,dataC(201:400),x,yp(201:400));
legend('Simulated data','Predicted data');