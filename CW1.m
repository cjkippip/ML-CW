clear;
clc;
m1=[0;3];
C1=[2 1;1 2];
m2=[4;0];
C2=[1 0;0 1];
N = 100;
X1=mvnrnd(m1,C1,N);
X2=mvnrnd(m2,C2,N);
%%
numGrid = 50;
xRange = linspace(-6.0, 6.0, numGrid);
yRange = linspace(-6.0, 6.0, numGrid);
P1 = zeros(numGrid, numGrid);
P2 = P1;
for i=1:numGrid
    for j=1:numGrid;
        x = [yRange(j) xRange(i)]';
        P1(i,j) = mvnpdf(x', m1', C1)/(mvnpdf(x', m1', C1)+mvnpdf(x', m2', C2));
        P2(i,j) = mvnpdf(x', m2', C2)/(mvnpdf(x', m1', C1)+mvnpdf(x', m2', C2));
        %分子分母同时约掉了P[w1]=P[w2]=0.5   P[w1]/P[w1]+P[w2]
    end
end
%%
figure(1),clf,
plot(X1(:,1),X1(:,2),'B+',X2(:,1),X2(:,2),'R+');
hold on;
u=C1^-1-C2^-1;
v=2*(C2^-1*m2-C1^-1*m1);
b1=m1'*C1^-1*m1-m2'*C2^-1*m2+log(det(C1)/det(C2));
h=ezplot(@(x,y) [x y]*u*[x y]'+[x y]*v+b1,[-6,8,-6,8]);
set(h,'Color','k', 'LineWidth', 2);
grid on;
hold off;
%%
figure(2),clf,
mesh(xRange,yRange,P1);
hold on;
mesh(xRange,yRange,P2);
axis([-6 6 -6 6]);
hold off;
%%
X1=[X1 ones(N,1)];
X2=[X2 ones(N,1)];
X=[X1; X2]';
T=[ones(N,1); -ones(N,1)]';
numHid=100;
[net]=feedforwardnet(numHid);
% net.divideParam.trainRatio=0.8;
% net.divideParam.valRatio=0.1;
% net.divideParam.testRatio=0.1;
[net]=train(net,X,T);
[output]=net(X);
%%
numGrid2 = 60;
xRange2 = linspace(-6.0, 8.0, numGrid2);
yRange2 = linspace(-6.0, 8.0, numGrid2);
emp=ones(2,numGrid2^2);
for i=1:numGrid2
    for j=1:numGrid2;
        xx = [xRange2(j) yRange2(i)]';
        emp(:,60*(i-1)+j)=xx;
    end
end
output2=net(emp);
grid1=emp(:,find(output2>0));
grid2=emp(:,find(output2<0));

figure(3),clf,
plot(grid1(1,:),grid1(2,:),'r.');
hold on
plot(grid2(1,:),grid2(2,:),'b.');
plot(X1(:,1),X1(:,2),'B+',X2(:,1),X2(:,2),'R+');
h=ezplot(@(x,y) [x y]*u*[x y]'+[x y]*v+b1,[-6,8,-6,8]);
set(h,'Color','k', 'LineWidth', 2);
title([num2str(numHid),' hidden nodes'], 'FontSize', 14);
xlabel('x', 'FontSize', 14);
ylabel('y', 'FontSize', 14);
grid on
hold off




