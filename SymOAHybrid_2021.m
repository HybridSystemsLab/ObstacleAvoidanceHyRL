clear all; close all; clc
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
addpath('AgentsOA2021')
load('Agent4298_FinalPi0')
critic0 = getCritic(saved_agent);
params0 = getLearnableParameterValues(critic0);
addpath('AgentsOA2021')
load('Agent4261_FinalPi1')
critic1 = getCritic(saved_agent);
params1 = getLearnableParameterValues(critic1);
load('Agent4357_final')
critic = getCritic(saved_agent);
params = getLearnableParameterValues(critic);
%%
close all
                
n = 60+1;           % Steps
dt = 0.05;          % Timestep
xi = [0, 0.15;
     0, 0.03;
     0, -0.1;
     0.3, 0];       % Initial conditions
xih = xi;
 
eps = 0.091;        % Noise
stop = false;
seed = 2;

 % Extended sets
X0 = [0.2000,    0.5980,   -0.3000,    0.2960];
X1 = [0.2020,    0.6000,   -0.4960,    0.1000];

figure

for ZZ = 1:length(xi)
    rng(seed);
    II = n;
    IIh = n;
    q = 1;          % Initialize logic parameter of the hybrid system
    for ii = 1:n
        dist(ii) = eps*(1 - 2*rand());
        
        % DQN Policy
        x(ii) = xi(ZZ,1);
        y(ii) = xi(ZZ,2)+dist(ii);
        d_ob = sqrt((x(ii)-1.5)^2+(y(ii))^2)-0.75;
        if d_ob < 0
            d_ob = 0;
        end
        d_go = sqrt((3-x(ii))^2+y(ii)^2);
        u = Findu_2_2021(params,[d_ob; d_go; y(ii)]);
        xi(ZZ,:) = xi(ZZ,:) + dt*[1, u]; 
        if (xi(ZZ,1) > 3 || abs(xi(ZZ,2)) >= 1.5 || d_ob == 0) && II > ii
            II = ii;
        end
        
        % HDQN Policy
        xh(ii) = xih(ZZ,1);
        yh(ii) = xih(ZZ,2)+dist(ii);
        d_obh = sqrt((xh(ii)-1.5)^2+(yh(ii))^2)-0.75;
        if d_obh < 0
            d_obh = 0;
        end
        d_goh = sqrt((3-xh(ii))^2+yh(ii)^2);
        
        if q == 0 && ((yh(ii)<=0 && xh(ii)>X0(2)) || (yh(ii)<=0 && xh(ii)<X0(1)) || yh(ii)<X0(3)) % X0
            q = 1;
        elseif q == 1 && ((yh(ii)>=0 && xh(ii)>X1(2)) || (yh(ii)>=0 && xh(ii)<X1(1)) || yh(ii)>X1(4)) % X1
            q = 0;
        end
        
        if q == 0
            uh = Findu_2_2021(params0,[d_obh; d_goh; yh(ii)]);
        else
            uh = Findu_2_2021(params1,[d_obh; d_goh; yh(ii)]);
        end
        
        xih(ZZ,:) = xih(ZZ,:) + dt*[1, uh]; 

        if (xih(ZZ,1) > 3 || abs(xih(ZZ,2)) >= 1.5 || d_obh == 0) && IIh > ii
            IIh = ii;
        end
    end
    line1 = plot(x(1:II),y(1:II),'LineWidth',2,'LineStyle',':','Color','red');
    hold on
    line2 = plot(xh(1:IIh),yh(1:IIh),'LineWidth',2,'LineStyle',':','Color','blue');
    plot(x(1),y(1),'o','Color','black','LineWidth',2,'MarkerSize',6)
    plot(x(II),y(II),'x','Color','red','LineWidth',2,'MarkerSize',10)
    plot(xh(IIh),yh(IIh),'x','Color','blue','LineWidth',2,'MarkerSize',10)
    xx = x(1:II);
    yy = y(1:II);
    
end

pgon1 = polyshape([2.95 2.95 3.05 3.05],[0.1,-.1 -.1 .1]);
hold on
grid on
xlim([0 3])
ylim([-1.55 1.55])
xlabel('$x$','FontSize',16,'interpreter','latex')
ylabel('$y$','FontSize',16,'interpreter','latex')

Z1 = linspace(0,3.5,100);
h1 = plot(pgon1,'FaceColor','blue');
C = [1.5,0, 1.1];
R = 0.75 ;    
theta=0:0.01:2*pi ;
xc=C(1)+R*cos(theta);
yc=C(2)+R*sin(theta);
zc = C(3)+zeros(size(xc)) ;
h4 = plot(xc, yc,'-','Color','black','LineWidth',2);
h2 = plot(Z1,ones(100,1)*1.5,'-','Color','black','LineWidth',2);
h3 = plot(Z1,ones(100,1)*-1.5,'-','Color','black','LineWidth',2);   