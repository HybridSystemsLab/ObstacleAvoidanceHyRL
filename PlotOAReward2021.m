clear all; close all; clc
resx = 500;
resy = 251;
x = linspace(0,3,resx);
y = linspace(-1.5,1.5,resy);

obx = 1.5;
oby = 0;
delta = .75;

d_up = 1-abs(y);
d_oby = abs(y)-0.25;
for ii = 1:resx
	if x(ii) < 1
		d_obx(ii) = 1-x(ii);
	elseif x(ii) < 2
		d_obx(ii) = 0;
	else
		d_obx(ii) = x(ii)-2;
	end
end

d_ob = NaN(resy,resx);
d_f  = NaN(resy,resx);
d_go = NaN(resy,resx);
d_upb = NaN(resy,resx);
R = NaN(resy,resx);
for ii = 1:resx
    for II = 1:resy
        d_ob(II,ii) = sqrt((x(ii)-obx)^2+(y(II)-oby)^2)-delta;
            if d_ob(II,ii) < 0
                    d_ob(II,ii) = 0;
            end
        Barrier(II,ii) = (d_ob(II,ii)-2*delta)^2-log(d_ob(II,ii));
        d_f(II,ii) = min(d_ob(II,ii),d_up(II));
        d_go(II,ii) = sqrt((3-x(ii))^2+y(II)^2);
        d_upb(II,ii) = d_up(II);
        R(II,ii) = -d_go(II,ii)-0.1*Barrier(II,ii)+3.5;
    end
    [Mi(ii), IndMi] = min(Barrier(:,ii));
    Ymi(ii) = y(IndMi);
    [Ma(ii), IndMa] = max(R(:,ii));
    Yma(ii) = y(IndMa);
end
figure
mesh(x,y,d_ob)
xlabel('$x$','FontSize',16,'interpreter','latex')
ylabel('$y$','FontSize',16,'interpreter','latex')
zlabel('$d_{ob}$','FontSize',16,'interpreter','latex')
hold on
% view(2)

figure
mesh(x,y,Barrier)
hold on
plot3(x,Ymi,Mi,'x','Color','red')
% view(90,0)
xlabel('$x$','FontSize',16,'interpreter','latex')
ylabel('$y$','FontSize',16,'interpreter','latex')
zlabel('Barrier','FontSize',16,'interpreter','latex')

figure
mesh(x,y,R)
hold on
plot3(x,Yma,Ma,'x','Color','red')
view(2)
xlabel('$x$','FontSize',16,'interpreter','latex')
ylabel('$y$','FontSize',16,'interpreter','latex')
zlabel('$R$','FontSize',16,'interpreter','latex')