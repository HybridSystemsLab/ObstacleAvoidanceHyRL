clear all; close all; clc
addpath('AgentsOA2021')
load('Agent4357_final')
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
critic = getCritic(saved_agent);
params = getLearnableParameterValues(critic);
%%
close all

n = 5;          % Number of simulation steps
dt = 0.05;      % Sampling time
N = 100;        % Number of simulations
T = n*dt;       % Horizon
rng(2);

qM0 = zeros(n,2);
maxqMeasuredM0 = zeros(n,2);
minqMeasuredM0 = zeros(N,2);
qM1 = zeros(n,2);
qMeasuredM1 = zeros(n,2);
maxqMeasuredM1 = zeros(N,2);
minqMeasuredM1 = zeros(N,2);

q0M0 = [linspace(0.4,0.6,N).' linspace(-0.1,0.3,N).'];     % Initial conditions for M_0
q0M1 = [linspace(0.4,0.6,N).' linspace(0.1,-0.3,N).'];     % Initial conditions for M_1

figure
for ZZ = 1:N
qM0 = [q0M0(randi(N),1); q0M0(randi(N),2)];
qM1 = [q0M1(randi(N),1); q0M1(randi(N),2)];
II = n;
    for ii = 1:n
        xM0(ii) = qM0(1);
        yM0(ii) = qM0(2);
        d_obM0 = sqrt((xM0(ii)-1.5)^2+(yM0(ii))^2)-0.75;
        if d_obM0 < 0
            d_obM0 = 0;
        end
        d_goM0 = sqrt((3-xM0(ii))^2+yM0(ii)^2);
        if (qM0(1) >= 0.4 && qM0(2) >=-0.1) || qM0(2) >= 0
            uM0 = Findu_2_2021(params,[d_obM0; d_goM0; yM0(ii)]);
        end
        qM0 = qM0 - dt*[1, uM0]; 
        if qM0(1) < 0 || abs(qM0(2)) >= 1.5 || d_obM0 == 0 
            II = ii;
            break
        end
    end
    maxqMeasuredM0(ZZ,:) = [max(xM0); max(yM0)];
    minqMeasuredM0(ZZ,:) = [min(xM0); min(yM0)];
    for ii = 1:n
        xM1(ii) = qM1(1);
        yM1(ii) = qM1(2);
        d_obM1 = sqrt((xM1(ii)-1.5)^2+(yM1(ii))^2)-0.75;
        if d_obM1 < 0
            d_obM1 = 0;
        end
        d_goM1 = sqrt((3-xM1(ii))^2+yM1(ii)^2);
        if (qM1(1) >= 0.4 && qM1(2) <=0.1) || qM1(2) <= 0
            uM1 = Findu_2_2021(params,[d_obM1; d_goM1; yM1(ii)]);
        end
        qM1 = qM1 - dt*[1, uM1]; 
        if qM1(1) < 0 || abs(qM1(2)) >= 1.5 || d_obM1 == 0 
            II = ii;
            break
        end
    end
    maxqMeasuredM1(ZZ,:) = [max(xM1); max(yM1)];
    minqMeasuredM1(ZZ,:) = [min(xM1); min(yM1)];
    line1 = plot(xM0(1:II),yM0(1:II),'LineWidth',0.1,'LineStyle','-','Color','red');
    hold on
    line2 = plot(xM1(1:II),yM1(1:II),'LineWidth',0.1,'LineStyle','-','Color','blue');
end
X0 = [min(minqMeasuredM0(:,1)), max(maxqMeasuredM0(:,1)), min(minqMeasuredM0(:,2)), max(maxqMeasuredM0(:,2))]
X1 = [min(minqMeasuredM1(:,1)), max(maxqMeasuredM1(:,1)), min(minqMeasuredM1(:,2)), max(maxqMeasuredM1(:,2))]
pgonX0 = polyshape([X0(1) X0(1) X0(2) X0(2)],[X0(4) X0(3) X0(3) X0(4)]);
pgonX1 = polyshape([X1(1) X1(1) X1(2) X1(2)],[X1(4) X1(3) X1(3) X1(4)]);
pgon1 = polyshape([2.95 2.95 3.05 3.05],[0.1,-.1 -.1 .1]);
hold on
grid on
xlim([0 3])
ylim([-1.55 1.55])
xlabel('$x$','FontSize',16,'interpreter','latex')
ylabel('$y$','FontSize',16,'interpreter','latex')
Z1 = linspace(0,3.5,100);
h1 = plot(pgon1,'FaceColor','blue');
hX0 = plot(pgonX0,'FaceColor','red');
hX1 = plot(pgonX1,'FaceColor','blue');
C = [1.5,0, 1.1] ;  
R = 0.75 ;  
theta=0:0.01:2*pi ;
xc=C(1)+R*cos(theta);
yc=C(2)+R*sin(theta) ;
zc = C(3)+zeros(size(xc)) ;
h4 = plot(xc, yc,'-','Color','black','LineWidth',2);
h2 = plot(Z1,ones(100,1)*1.5,'-','Color','black','LineWidth',2);
h3 = plot(Z1,ones(100,1)*-1.5,'-','Color','black','LineWidth',2);   