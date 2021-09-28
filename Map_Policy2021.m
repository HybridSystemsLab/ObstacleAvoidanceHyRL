clear all; close all; clc
addpath('AgentsOA2021')
load('Agent4357_final')
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
%%
critic = getCritic(saved_agent);
params = getLearnableParameterValues(critic);
resx = 300;
resy = resx;
x = linspace(0,3,resx);
y = linspace(-1.5,1.5,resy);
U = NaN(resy,resx);
Um = zeros(resx,resy);
Qm = zeros(resx,resy);

for II = 1:resy
    d_oby = abs(y(II))-0.75;
    d_ub = 1-abs(y(II));
    for ii = 1:resx
        d_ob = sqrt((x(ii)-1.5)^2+(y(II))^2)-0.75;
            if d_ob < 0
                    d_ob = 0;
            end
            d_go = sqrt((3-x(ii))^2+y(II)^2);
        U(II,ii) = Findu_2_2021(params,[d_ob; d_go; y(II)]);
    end
end
%% Plotting
close all; 
figure
mesh(x,y,U)
view(2)
grid on
hold on 
h = colorbar('Ticks',[-1 -.5 0 .5 1]);
colormap(parula(5));
C = [1.5,0, 1.05] ;   
R = 0.75 ;    
theta=0:0.01:2*pi ;
xc=C(1)+R*cos(theta);
yc=C(2)+R*sin(theta) ;
zc = C(3)+zeros(size(xc)) ;
patch(xc,yc,zc,'black')
xlabel('$x$','FontSize',16,'interpreter','latex')
ylabel('$y$','FontSize',16,'interpreter','latex')
ylabel(h, '$u$','FontSize',16,'interpreter','latex')
% saveas(gcf,'PolicyMap_ObstacleAvoid_CriticPoints','epsc')