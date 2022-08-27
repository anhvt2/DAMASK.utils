clear all
load data1

theta=[10 10];
lob=[1e-1 1e-1];
upb=[20 20];
% [dmodel, perf]=dacefit(S,Y,@regpoly0,@corrgauss,theta,lob,upb);
[dmodel, perf]=dacefit(S,Y,@regpoly0,@corrgauss,theta,lob,upb);
% X=[1 2];
% X=gridsamp([0 0;100 100],40);
X=[1 3;2 3;7 8];
% X=gridsamp([0 0;2 2],[2 3])
[YX,MSE]=predictor(X,dmodel);
% X1=reshape(X(:,1),40,40);
% X2=reshape(X(:,2),40,40);
% YX=reshape(YX,size(X1));
% figure(1), mesh(X1,X2,YX)
% hold on,
% plot3(S(:,1),S(:,2),Y,'.k','MarkerSize',10)
% hold off
% [emodel perf]=dacefit(S,Y,@regpoly0,@correxp,2);
% figure(2), mesh(X1,X2,reshape(MSE,size(X1)))
% [y, dy]=predictor(S(1,:),dmodel);