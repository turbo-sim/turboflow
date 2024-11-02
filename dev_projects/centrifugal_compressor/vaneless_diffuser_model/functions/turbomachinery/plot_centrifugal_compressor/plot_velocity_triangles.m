function [] = plot_velocity_triangles_A(turbine_data,my_filename,my_path,save)

% Plot the velocity triangles (option A)

% Common origin for the blade velocity (convenient for axial turbines)
% The location of the labels for each vector might not be perfect
% The location of the labels can be tweaked manually if a high quality
% figure is desired
% Just save again the figure after tweaking the location of the labels

% Import the velocity triangles
u =[turbine_data.plane.u];
v_t = [turbine_data.plane.v_t];
v_m = [turbine_data.plane.v_m];
w_t = [turbine_data.plane.w_t];
w_m = [turbine_data.plane.w_m];
% Ma_rel = [turbine_data.plane.Ma_rel];
% Ma = [turbine_data.plane.Ma];

% Rename the variables
v_3t = v_t(3:4:end-1);
v_4t = v_t(4:4:end);
v_3m = v_m(3:4:end-1);
v_4m = v_m(4:4:end);
w_3t = w_t(3:4:end-1);
w_4t = w_t(4:4:end);
w_3m = w_m(3:4:end-1);
w_4m = w_m(4:4:end);
u_3 = u(3:4:end-1);
u_4 = u(4:4:end);
% Ma_rel3 = Ma_rel(3:4:end-1);
% Ma_rel4 = Ma_rel(4:4:end);
% Ma_3 = Ma(3:4:end-1);
% Ma_4 = Ma(4:4:end);

% Prepare the figure
fig = figure();
n_stages = turbine_data.overall.n_stages;

for j = 1:n_stages
    
    % Prepare the subfigure
    subplot(n_stages,1,j)
    hold on; axis image; axis off;
    axis([min([0 v_3t,v_4t]) 1.1*max([v_3t,v_4t,u_3,u_4]) -max([v_3m,v_4m]) 0])
    
    % Plot the velocity vectors
    Length = 12;
    width = 0.5;
    b_ang = 50;         % Base angle
    t_ang = 15;         % Tip angle
    
    arrow([0 0],[u_3(j) 0],Length,'BaseAngle',b_ang,'TipAngle',t_ang,'LineWidth',width)
    arrow([u_3(j) 0],[u_3(j)+w_3t(j) -w_3m(j)],Length,'BaseAngle',b_ang,'TipAngle',t_ang,'LineWidth',width)
    arrow([0 0],[v_3t(j) -v_3m(j)],Length,'BaseAngle',b_ang,'TipAngle',t_ang,'LineWidth',width)
    arrow([u_3(j) 0],[u_3(j)+w_4t(j) -w_4m(j)],Length,'BaseAngle',b_ang,'TipAngle',t_ang,'LineWidth',width)
    arrow([0 0],[v_4t(j) -v_4m(j)],Length,'BaseAngle',b_ang,'TipAngle',t_ang,'LineWidth',width)

    % Label the velocity vectors
    fontsize = 11;
%     % Do not display mach number
%     string = '$\quad v_{1}$'; text(c_1t,-0.95*c_x,string,'HorizontalAlignment','left','FontSize',fontsize);
%     string = '$v_{3} \quad$'; text(v_3t(j),-1.125*v_3m(j),string,'HorizontalAlignment','right','FontSize',fontsize);
%     string = '$\quad w_{3}$'; text(w_3t(j)/2+u_3(j),-0.45*w_3m(j),string,'HorizontalAlignment','left','FontSize',fontsize);
%     string = '$v_{4} \quad$'; text(v_4t(j)/2,-0.45*v_4m(j),string,'HorizontalAlignment','right','FontSize',fontsize);
%     string = '$\quad w_{4}$'; text(w_4t(j)+u_3(j),-1.125*w_4m(j),string,'HorizontalAlignment','left','FontSize',fontsize);
%     string = '$u$'; text(u_3(j)/2,-0.125*v_3m(j),string,'HorizontalAlignment','right','FontSize',fontsize);

    % Display Mach number in parenthesis
    string = ['$v_{3} \, (',num2str(Ma_3(j),'%.2f'),') \quad$']; text(v_3t(j),-1.125*v_3m(j),string,'HorizontalAlignment','right','FontSize',fontsize);
    string = ['\quad','$w_{3} \, (',num2str(Ma_rel3(j),'%.2f'),')$']; text(w_3t(j)/2+u_3(j),-0.45*w_3m(j),string,'HorizontalAlignment','left','FontSize',fontsize);
    string = ['$v_{4} \, (',num2str(Ma_4(j),'%.2f'),') \quad$']; text(v_4t(j)/2,-0.45*v_4m(j),string,'HorizontalAlignment','right','FontSize',fontsize);
    string = ['\quad','$w_{4} \, (',num2str(Ma_rel4(j),'%.2f'),')$']; text(w_4t(j)+u_3(j),-1.125*w_4m(j),string,'HorizontalAlignment','left','FontSize',fontsize);
    string = '$u$'; text(u_3(j)/2,-0.125*v_3m(j),string,'HorizontalAlignment','right','FontSize',fontsize);
    
end

% Save the figure
name = 'triangles_A';
if save == 1
    saveas(fig,fullfile(my_path,[my_filename,'_',name,'.pdf']),'pdf')
elseif save == 2
%     saveas(fig,fullfile(my_path,[my_filename,'_',name]),'fig')
%     export_fig(fig,fullfile(my_path,[my_filename,'_',name]),'-png','-r1000')
%     export_fig(fig,fullfile(my_path,[my_filename,'_',name]),'-eps','-painters')
    export_fig(fig,fullfile(my_path,[my_filename,'_',name]),'-pdf','-painters')
elseif save ~= 0
    error('Choose a valid saving option')
end

end
