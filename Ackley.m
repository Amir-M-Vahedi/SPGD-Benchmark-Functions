clc;clear;close all;
rng(123)
rng(143)
rng(343)
rng(53)
rng(234)
% rng(53)
%% Gradient Descent
lr=.001;
tel=1e-6;
Iter_max=500;
% x_g=[3,3];
x_g0=4*(2*rand(1,2)-1);
% x_g0=[0,1.5]
xb_g=x_g0;
x_g=x_g0;
O_g=Object(x_g(1),x_g(2));
Op_g=Object_prime(x_g(1),x_g(2));
Iter=1;
x_g_History=[x_g,O_g];
tic
while Iter<=Iter_max & norm(Op_g)>=tel
    x_g= x_g - lr * Op_g;
    Otemp_g= Object(x_g(1),x_g(2));
    x_g_History(Iter+1,:)=[x_g,Otemp_g];
    if Otemp_g<O_g
        O_g= Otemp_g;
        xb_g = x_g;
    end
    %Calculate the derivative
    Op_g=Object_prime(x_g(1),x_g(2));
    Iter= Iter+1;
end
T_g=toc
fcount_g=Iter;
%% IPGD
fcount_gsa=0;
N_rrt=10;
Iter_p=5;
amp=2.5;
xb_gsa=x_g0;
x_gsa=x_g0;
O_gsa=Object(x_gsa(1),x_gsa(2));
fcount_gsa= fcount_gsa+1;
Op_gsa=Object_prime(x_gsa(1),x_gsa(2));
Iter=1;
Indx=1;
x_gsa_History=[x_gsa,O_gsa];
tic
while Iter<=Iter_max & norm(Op_gsa)>=tel
    x_gsa= x_gsa - lr * Op_gsa;
    Otemp_gsa= Object(x_gsa(1),x_gsa(2));
    fcount_gsa= fcount_gsa+1;
    x_gsa_History(Indx+1,:)=[x_gsa,Otemp_gsa];
    Indx=Indx+1;
    if Otemp_gsa<O_gsa
        O_gsa= Otemp_gsa;
        xb_gsa = x_gsa;
    end
    if mod(Iter,Iter_p)==0
        % Perturb 1st Variable
        test_list=x_gsa.*ones(N_rrt,2);
        O_test=[];
        for i=1:N_rrt
            test_list(i,:)=test_list(i,:)+ amp*(2*rand(1,2)-1);
            temp=Object(test_list(i,1),test_list(i,2));
            fcount_gsa= fcount_gsa+1;
            O_test=[O_test, temp];
        end
        [O_tet_min,i_min]=min(O_test);
        if O_tet_min<=Otemp_gsa
            x_gsa = test_list(i_min,:);
            Otemp_gsa= O_tet_min;
            x_gsa_History(Indx+1,:)=[x_gsa,Otemp_gsa];
            Indx=Indx+1;
            if Otemp_gsa<O_gsa
                O_gsa= Otemp_gsa;
                xb_gsa = x_gsa;
            end
        end
    end
    %Calculate the derivative
    Op_gsa=Object_prime(x_gsa(1),x_gsa(2));
    Iter= Iter+1;
end
T_gsa=toc
%% Perturbed Gradient Descent
fcount_pg=0;
N_rrt=5;
amp=2.5;
xb_pg=x_g0;
x_pg=x_g0;
xnoise_pg=x_pg;
O_pg=Object(x_pg(1),x_pg(2));
fcount_pg= fcount_pg +1;
Onoise_pg=O_pg;
Op_pg=Object_prime(x_pg(1),x_pg(2));
Iter=1;
Indx=1;
x_pg_History=[x_pg,O_pg];
Iter_thershold=100;
Iter_noise=-Iter_thershold-1;

tic
while Iter<=Iter_max
    x_pg= x_pg - lr * Op_pg;
    Otemp_pg= Object(x_pg(1),x_pg(2));
    fcount_pg= fcount_pg +1;
    x_pg_History(Indx+1,:)=[x_pg,Otemp_pg];
    Indx=Indx+1;
    if Otemp_pg<O_pg
        O_pg= Otemp_pg;
        xb_pg = x_pg;
    end

    if abs(norm(Op_pg))<=1000*tel && Iter-Iter_noise>Iter_thershold
        Iter_noise=Iter;
        Onoise_pg= Otemp_pg;
        xnoise_pg=x_pg;
        x_pg= x_pg + amp*(2*rand(1,2)-1);
        Otemp_pg= Object(x_pg(1),x_pg(2));
        fcount_pg= fcount_pg +1;
        x_pg_History(Indx+1,:)=[x_pg,Otemp_pg];
        Indx=Indx+1;
        if Otemp_pg<O_pg
            O_pg= Otemp_pg;
            xb_pg = x_pg;
        end
    end
    if Iter-Iter_noise==Iter_thershold
        if Otemp_pg > Onoise_pg
            Otemp_pg = Onoise_pg;
            O_pg= Otemp_pg;
            xb_pg = xnoise_pg;
            break
        end
    end
    %Calculate the derivative
    Op_pg=Object_prime(x_pg(1),x_pg(2));
    Iter= Iter+1;
end
T_pg=toc
%% SA
% T0=1000;
% T_N=100;
% Temp=T0;
% amp=2.5;
% D=amp*eye(2);
% alpha=0.1;
% xb_sa=x_g0;
% x_sa=x_g0;
% O_sa=Object(x_sa(1),x_sa(2));
% Iter=1;
% Indx=1;
% x_sa_History=[x_sa,O_sa];
% tic
% while Iter<=Iter_max
%     xtemp_sa= x_sa + (D*(2*rand(1,2)-1)')';
%     Otemp_sa= Object(xtemp_sa(1),xtemp_sa(2));
%     x_sa_History(Indx+1,:)=[xtemp_sa,Otemp_sa];
%     Indx=Indx+1;
%     if Otemp_sa<O_sa
%         O_sa= Otemp_sa;
%         xb_sa = xtemp_sa;
%         D= (1-alpha)*D + alpha * abs(x_sa-xtemp_sa)'.*eye(2);
%         x_sa = xtemp_sa;
%     end
%     if rand<exp(-(Otemp_sa-O_sa)/Temp)
%         D= (1-alpha)*D + alpha * abs(x_sa-xtemp_sa)'.*eye(2);
%         x_sa = xtemp_sa;
%     end
%     Temp=T0-i*(T0-T_N)/Iter_max;
%     Iter= Iter+1;
% end
% toc

lb = [-4,-4];
ub = [4,4];

options = optimoptions(@simulannealbnd,'MaxIter',Iter_max,'OutputFcns',@saoutfun);
% options.Display='iter';
tic
[xbsa,fvalsa,~,outputsa]  = simulannealbnd(@ObjectSA,x_g0,lb,ub,options);
T_sa=toc

%% Genetic
% options = optimoptions('ga','Display','iter','MaxGenerations',Iter_max,'OutputFcns',@gaoutfun);
% lb = [-4,-4];
% ub = [4,4];
% [xga,fvalga,exitflag,outputga]  = ga(@ObjectSA,2,[],[],[],[],lb,ub,[],[],options)

%% Fmincon
fun = @ObjectMinco;
x0 = x_g0;
A = [];
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];
nonlcon = [];
options = optimoptions('fmincon','SpecifyObjectiveGradient',true,'OutputFcn',@outfun);
% options.Display='iter';
tic
[Sol_Fmincon,x_mincon,history_mincon,history_mincon2] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);
T_fmincon=toc
Fmincon_b=[Sol_Fmincon, Object(Sol_Fmincon(1),Sol_Fmincon(2))];

%% FminSearch
Sol_FminSearch = fminsearch(fun,x0);
FminSearch_b=[Sol_FminSearch, Object(Sol_FminSearch(1),Sol_FminSearch(2))];

%% Fminunc
options2 = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
% options2.Display='iter';
Sol_Fminunc = fminunc(fun,x0,options2);
Fminunc_b=[Sol_Fminunc, Object(Sol_Fminunc(1),Sol_Fminunc(2))];
%% Visualization of The problem
D_max=5;
n=300;
x= linspace(-D_max,D_max,n);
y= linspace(-D_max,D_max,n);
[X,Y]=meshgrid(x,y);
Z= F(X,Y);
% s=mesh(X,Y,Z);
fig = figure;
set(fig, 'Units', 'inches', 'Position', [0, 0, 9, 8]); % Width=6in, Height=4in
s=surf(X,Y,Z,'EdgeColor','none','FaceAlpha',0.8);
% s.FaceColor = 'flat';
xlabel('x')
ylabel('y')
zlabel('f')
colorbar()
[~,i]=min(Z,[],'all');
hold on
% contour(X,Y,Z,10)
% Best_sol=[X(i),Y(i),Z(i)];
Best_sol=[0,0,Object(0,0)];
f0=plot3(x_g0(1),x_g0(2),F(x_g0(1),x_g0(2)),'.','MarkerEdgeColor',"#7E2F8E","MarkerSize",20, 'DisplayName', 'Start Point');
f1=plot3(Best_sol(1),Best_sol(2),Best_sol(3),'g.',"MarkerSize",20, 'DisplayName', 'Optimal Solution');

plot3(xb_g(1),xb_g(2),O_g,'b.',"MarkerSize",15)
f2=plot3(x_g_History(:,1),x_g_History(:,2),x_g_History(:,3),'b-',"LineWidth",2, 'DisplayName', 'GD');

plot3(xb_gsa(1),xb_gsa(2),O_gsa,'r.',"MarkerSize",15)
f3=plot3(x_gsa_History(:,1),x_gsa_History(:,2),x_gsa_History(:,3),'r.-',"LineWidth",2, 'DisplayName', 'SPGD');

% plot3(xb_sa(1),xb_sa(2),O_sa,'m.',"MarkerSize",20)
% f4=plot3(x_sa_History(:,1),x_sa_History(:,2),x_sa_History(:,3),'m-',"LineWidth",2, 'DisplayName', 'SA');
% plot3(xbsa(1),xbsa(2),fvalsa,'m.',"MarkerSize",20)
% f4=plot3(history_sa.x(:,1),history_sa.x(:,2),history_sa.fval(:),'m-',"LineWidth",2, 'DisplayName', 'SA');


% f5=plot3(Fmincon_b(1),Fmincon_b(2),Fmincon_b(3),'k*',"MarkerSize",10, 'DisplayName', 'Fminco');
% f6=plot3(FminSearch_b(1),FminSearch_b(2),FminSearch_b(3),'y*',"MarkerSize",10, 'DisplayName', 'FminSearch');
% f7=plot3(Fminunc_b(1),Fminunc_b(2),Fminunc_b(3),'c*',"MarkerSize",10, 'DisplayName', 'Fminunc');

plot3(xb_pg(1),xb_pg(2),O_pg,'m.',"MarkerSize",15)
f8=plot3(x_pg_History(:,1),x_pg_History(:,2),x_pg_History(:,3),'m-.',"LineWidth",2, 'DisplayName', 'PGD');

plot3(Fmincon_b(1),Fmincon_b(2),Fmincon_b(3),'k.',"MarkerSize",15)
f9=plot3(optimhistory.x(:,1),optimhistory.x(:,2),optimhistory.fval,'k-',"LineWidth",2, 'DisplayName', 'Fmincon');

% legend([f1,f2,f3,f8])
lgd = legend([f0, f1,f2,f3,f8,f9]);
set(lgd, 'FontSize', 10);
% legend([f1,f2,f3,f4,f8])
% legend([f1,f2,f3,f4,f5,f6,f7])
% axis equal
% print('test.eps','-depsc')

fig = figure;
set(fig, 'Units', 'inches', 'Position', [0, 0, 12, 8]); % Width=6in, Height=4in
plot(x_g_History(:,3),'b',"LineWidth",1)
hold on
plot(x_pg_History(:,3),'m-.',"LineWidth",1)
plot(history_sa.fval,'Color','#EDB120',"LineWidth",1)
plot(optimhistory.fval,'k',"LineWidth",1)
plot(x_gsa_History(:,3),'r',"LineWidth",1)
xlabel('Iteration')
ylabel('f(x)')
lgd =legend('GD','PGD','SA','Fmincon','SPGD');
set(lgd, 'FontSize', 10);
disp("Time:")
T_g*1000
T_pg*1000
T_fmincon*1000
T_sa*1000
T_gsa*1000

disp("Iter:")
fcount_g
fcount_pg
fcount_fmincon=history_mincon2.funcCount
fcount_sa= outputsa.funccount
fcount_gsa

disp("Bests:")
O_g
O_pg
Fmincon_b(3)
fvalsa
O_gsa
%% Utilities
function z=F(X,Y)
z=-20.*exp(-0.2.*sqrt(0.5.*(X.^2+Y.^2)))-exp(0.5.*(cos(2.*pi.*X)+cos(2.*pi.*Y)))+exp(1)+20;
end

function z=Object(x,y)
z=-20*exp(-0.2*sqrt(0.5*(x^2+y^2)))-exp(0.5*(cos(2*pi*x)+cos(2*pi*y)))+exp(1)+20;
end

function z=Object_prime(x,y)
z=[pi*exp((cos(2*pi*x)/2 + cos(2*pi*y)/2))*sin(2*pi*x) + (2*x*exp((-sqrt(x^2/2 + y^2/2)/5)))/sqrt(x^2/2 + y^2/2),...
   pi*exp((cos(2*pi*x)/2 + cos(2*pi*y)/2))*sin(2*pi*y) + (2*y*exp((-sqrt(x^2/2 + y^2/2)/5)))/sqrt(x^2/2 + y^2/2)];
end



function [f,g]=ObjectMinco(x)
f=-20*exp(-0.2*sqrt(0.5*(x(1)^2+x(2)^2)))-exp(0.5*(cos(2*pi*x(1))+cos(2*pi*x(2))))+exp(1)+20;

if nargout > 1 % gradient required
    g=[pi*exp((cos(2*pi*x(1))/2 + cos(2*pi*x(2))/2))*sin(2*pi*x(1)) + (2*x(1)*exp((-sqrt(x(1)^2/2 + x(2)^2/2)/5)))/sqrt(x(1)^2/2 + x(2)^2/2),...
   pi*exp((cos(2*pi*x(1))/2 + cos(2*pi*x(2))/2))*sin(2*pi*x(2)) + (2*x(2)*exp((-sqrt(x(1)^2/2 + x(2)^2/2)/5)))/sqrt(x(1)^2/2 + x(2)^2/2)];
end

end

function [f]=ObjectSA(x)
f=-20*exp(-0.2*sqrt(0.5*(x(1)^2+x(2)^2)))-exp(0.5*(cos(2*pi*x(1))+cos(2*pi*x(2))))+exp(1)+20;
end

function stop = outfun(x,optimValues,state)
     persistent history
     stop = false;
 
     switch state
         case 'init'
             history.x = [];
             history.fval = [];
         case 'iter'
         % Concatenate current point and objective function
         % value with history. in must be a row vector.
           history.fval = [history.fval; optimValues.fval];
           history.x = [history.x; x];
         case 'done'
             assignin('base','optimhistory',history);
         otherwise
     end
end

function [stop ,optnew , changed ] = saoutfun(x,optimValues,state)
     persistent history
     stop = false;
     changed = false;
     optnew = 0;
     switch state
         case 'init'
             history.x = [];
             history.fval = [];
         case 'iter'
         % Concatenate current point and objective function
         % value with history. in must be a row vector.
           history.fval = [history.fval; optimValues.fval];
           history.x = [history.x; optimValues.x];
         case 'done'
             assignin('base','history_sa',history);
         otherwise
     end
end

function [optimValues ,optnew , changed ] = gaoutfun(x,optimValues,state)
     persistent history
     optimValues.StopFlag = 'n';
     changed = false;
     optnew = 0;
     switch state
         case 'init'
             history.x = [];
             history.fval = [];
         case 'iter'
         % Concatenate current point and objective function
         % value with history. in must be a row vector.
           [fval,id]=min(optimValues.Score);
           history.fval = [history.fval; fval];
           history.x = [history.x; optimValues.Population(id,:)];
         case 'done'
             assignin('base','history_ga',history);
         otherwise
     end
 end