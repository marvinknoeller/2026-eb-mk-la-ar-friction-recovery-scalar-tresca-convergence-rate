close all
clear all
clc


load('order_finite_elementsNLnew.mat')

fig = figure();
set(gcf, 'Color', 'w')
fsize = 26;

loglog(h,errorH1,'^-','LineWidth',3,'MarkerSize',10, 'MarkerFaceColor','w', 'Color','#3B5BA5')
hold on
loglog(h,errorL2,'o-','LineWidth',3,'MarkerSize',10, 'MarkerFaceColor','w','Color','#E87A5D')

loglog(h,h.^2*1e0,'k--','LineWidth',3)
loglog(h,h*5e0,'k--','LineWidth',3)
lgd = legend({'$\Vert u - u_h \Vert_{H^1(\Omega)}$', '$\Vert u - u_h \Vert_{L^2(\Omega)}$'},'FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
ax = gca;
ax.FontSize = fsize;
ax.XLim = [1e-3, 2e-1];
ax.YLim = [1e-6, 1e1];
ax.GridAlpha = .4;
ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
ax.XLabel.String = '$h$';%_{\mathrm{max}}$';
ax.XLabel.Interpreter = 'latex';
ax.YTick = logspace(-5,0,6);

grid on

saveas(gcf,strcat('plots/','numerical_example'),'epsc')