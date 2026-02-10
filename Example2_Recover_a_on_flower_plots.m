clear all
close all
clc

cos_num = 6;
sin_num = 6;
load('2NLRec_6_6delta0pz.mat')
cos_coeffs_ex = [8.0, 1., -.5, 0.5, 0.1, -.7];
sin_coeffs_ex = [.8, .1, -1, 0.5, 0.2, .4];
snapshots = [1, 26, 101, size(coeff_history,2)];
[xx, vals_ex] = create_fun_for_plot(cos_coeffs_ex, sin_coeffs_ex);
fig = figure();
set(gcf, 'Color', 'w')
fsize = 36;
ylevel = -.7;
for kk = snapshots
    plot(xx,vals_ex,'--b','LineWidth',4);
    hold on
    plot(xx,vals_history(:,kk),'-k','LineWidth',4);
    hold off
    ax = gca;
    ax.XLim = [0,2*pi];
    ax.YLim = [1,20];
    ax.FontSize = fsize;
    ax.XTick = 0:pi/2:2*pi;
    
    ax.XTickLabel(1) = {'$0$'};
    ax.XTickLabel(2) = {'$\frac{\pi}{2}$'};
    ax.XTickLabel(3) = {'$\pi$'};
    ax.XTickLabel(4) = {'$\frac{3\pi}{2}$'};
    ax.XTickLabel(5) = {'$2\pi$'};
    ax.TickLabelInterpreter = 'LaTex';
    ax.GridAlpha = .9;
    ax.GridLineStyle = '--';
    if kk == 1
        lgd = legend('Exact $\tilde{a}$', strcat("Initial guess"),'FontSize',fsize, 'Interpreter','Latex','LineWidth',2,...
            'Position',[0.203392159366284 0.691904762813023 0.439465005057199 0.21642857052031]);
        lgd.ItemTokenSize = [50, 18];
    else
        lgd = legend('Exact $\tilde{a}$', strcat("Iteration ",num2str(kk-1)),'FontSize',fsize, 'Interpreter','Latex','LineWidth',2,...
            'Position',[0.203392159366284 0.691904762813023 0.439465005057199 0.21642857052031]);
        lgd.ItemTokenSize = [50, 18];
    end
    grid on
    drawnow
    saveas(gcf,strcat('plots/','ex_1_',num2str(kk)),'epsc')
end


function [xx, vals] = create_fun_for_plot(cos_coeffs, sin_coeffs)
    nn = 1000;
    xx = linspace(0,2*pi,nn);
    vals = zeros(1,nn);
    
    for nn = 1:length(cos_coeffs)
        vals = vals + 2/sqrt(pi)*cos_coeffs(nn) * cos((nn-1)*xx);
    end
    
    for nn = 1:length(sin_coeffs)
        vals = vals + 2/sqrt(pi)*sin_coeffs(nn) * sin(nn*xx);
    end
end