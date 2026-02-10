close all
clear all
clc


load('4NLorder_a.mat')

fig = figure();
set(gcf, 'Color', 'w')
fsize = 26;
err_vec = err_vec_C2/total_C2;
loglog(hh,err_vec,'^-','LineWidth',3,'MarkerSize',10, 'MarkerFaceColor','w', 'Color','#3B5BA5')
hold on
% loglog(hh,errorL2,'o-','LineWidth',3,'MarkerSize',10, 'MarkerFaceColor','w','Color','#E87A5D')

loglog(hh,hh.^2*6e3,'k--','LineWidth',3)
% loglog(hh,hh*5e0,'k--','LineWidth',3)
lgd = legend({'$e_\mathrm{rel}(h)$'},'FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
% lgd.Position = [0.502462768554688,0.25,0.377894374302455,0.089523808161418];
ax = gca;
ax.FontSize = fsize;
ax.XLim = [1e-3, 1.5e-2];
ax.YLim = [1e-3, 1e1];
ax.GridAlpha = .4;
ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
ax.XLabel.String = '$h$'; %_{\mathrm{max}}$';
ax.XLabel.Interpreter = 'latex';
% ax.YTick = logspace(-5,0,6);

grid on

saveas(gcf,strcat('plots/','order_aNL'),'epsc')