close all
clear all
clc


% load('4NLorder_a.mat')
load('NLorder_delta1e-07.mat')
fig = figure();
% fig.Position = [476 360 689 420];
set(gcf, 'Color', 'w')
fsize = 26;
err_vec = err_vec_C2/total_C2;
loglog(hh,err_vec,'<-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#71B379')
hold on

load('NLorder_delta3e-07.mat')
err_vec = err_vec_C2/total_C2;
loglog(hh,err_vec,'^-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#B25690')
 
% load('NLorder_delta5e-07.mat')
% err_vec = err_vec_C2/total_C2;
% loglog(hh,err_vec,'^-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#B25690')


load('NLorder_delta1e-06.mat')
err_vec = err_vec_C2/total_C2;
loglog(hh,err_vec,'d-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#EDC400')

load('NLorder_delta3e-06.mat')
err_vec = err_vec_C2/total_C2;
loglog(hh,err_vec,'o-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#1D71BA')

% load('NLorder_delta5e-06.mat')
% err_vec = err_vec_C2/total_C2;
% loglog(hh,err_vec,'o-','LineWidth',3,'MarkerSize',12, 'MarkerFaceColor','w','Color','#1D71BA')

loglog(hh,hh.^2*6e3,'k--','LineWidth',3)

lgd = legend('$\sigma = 1\times 10^{-7}$','$\sigma = 3\times10^{-7}$',...
    '$\sigma = 1\times10^{-6}$', '$\sigma = 3\times10^{-6}$','FontSize',fsize, 'Interpreter','Latex','LineWidth',2, ...
    'Location','SouthEast');
% lgd.Position = [0.61378263745989,0.250476191157389,0.261217362540109,0.341428565979004];
lgd.NumColumns = 2;
% lgd.Title.String = "noise level";
text(0.00308,0.0177,'$e_{\mathrm{rel}}^\delta(h)$ for noise levels','Interpreter','Latex','FontSize',fsize,'BackGroundColor','w','EdgeColor','k','LineWidth',1)
ax = gca;
ax.FontSize = fsize;
ax.XLim = [1e-3, 1.5e-2];
ax.YLim = [1e-3, 1e1];
ax.GridAlpha = .4;
ax.MinorGridAlpha = 0.2;
ax.GridLineStyle = '-';
ax.LineWidth = 2.0;
ax.TickLength = [0.02, 0.2];
% ax.XLabel.String = '$h_{\mathrm{max}}$';
ax.XLabel.String = '$h$';
ax.XLabel.Interpreter = 'latex';
% ax = gca;
% ax.FontSize = fsize;
% ax.YLim = [1e-4, 1e2];
% ax.GridAlpha = .8;
% ax.MinorGridAlpha = 0.2;
% ax.GridLineStyle = '-';
% ax.LineWidth = 2.0;
% ax.TickLength = [0.02, 0.2];
% ax.XLabel.String = 'h';
% ax.XLabel.Interpreter = 'latex';
grid on

saveas(gcf,strcat('plots/','ex_3_ordernoise'),'epsc')