%=========================================================================
%                                                                     
%       BIOMEDICAL IMAGING
%       MRI 1
%
%=========================================================================

%=========================================================================
%	Magnetization Dynamics
%=========================================================================

function [] = MAGNETIZATION_DYNAMICS()

    clear all; close all; 
    
    fprintf ( '-----------------------------------------\n' );  
    fprintf ( '2 MAGNETIZATION DYNAMICS                 \n' );  
    fprintf ( '-----------------------------------------\n' );  
    
    M0 = 1;                                                                             % Equilibrium magnetization
    T1 = 200;                                                                           % T1 [msec]
    T2 = 50;                                                                            % T2 [msec]
    angle = 38.85 * pi/180;                                                             % Flip angle [rad]
    TR = 50;                                                                            % Pulse interval (repetition time)[msec]
    np = 50;                                                                            % Number of pulses
    nt = 100;                                                                           % Number of time points to resolve relaxation between pulses
    
    Mz = zeros(np);                                                                     % To store Mz before each pulse 
    Mxy = zeros(np);                                                                    % To store Mxy after each pulse
    Mz_highres = zeros(nt,np);                                                          % To store course of Mz relaxation after each pulse
    Mxy_highres = zeros(nt,np);                                                         % To store course of Mxy relaxation after each pulse
    t = [0:(nt-1)].*TR./(nt-1);                                                         % Vector of time values covering one inter-pulse interval
        
    Mz_before_pulse = M0;                                                               % Initial condition
    Mxy_before_pulse = 0;                                                               % "
    
    for p = 1:np                                                                        % Loop through pulses
        
        Mz(p) = Mz_before_pulse;                                                        % Store Mz before RF pulse ("available magnetization")
        
        Mz_after_pulse = Mz_before_pulse*cos(angle) - Mxy_before_pulse*sin(angle);      % Rotation by RF pulse 
        Mxy_after_pulse = Mz_before_pulse*sin(angle) + Mxy_before_pulse*cos(angle);     % "
        
        Mxy(p) = Mxy_after_pulse;                                                       % Store Mxy after RF pulse ("transverse magnetization giving signal") 
 
        Mz_highres(:,p) = (Mz_after_pulse - M0) * exp(-t./T1) + M0;                     % Store course of Mz relaxation after RF pulse
        Mxy_highres(:,p) = Mxy_after_pulse * exp(-t./T2);                               % Store course of Mxy relaxation after RF pulse
        
        Mz_before_pulse = Mz_highres(nt,p);                                             % Relaxation
        Mxy_before_pulse = Mxy_highres(nt,p);                                           % "
        
    end
    
    figure                                                                              % Plot magnetization at RF pulse times                     
    subplot(2,1,1), plot(Mz), axis tight, ylim([-1,1]), title('Mz');
    subplot(2,1,2), plot(Mxy), axis tight, ylim([-1,1]), title('Mxy');
    
    figure                                                                              % Plot magnetization with high time resolution, including relaxation between RF pulses 
    subplot(2,1,1), plot(reshape(Mz_highres,1,np*nt)), axis tight, ylim([-1,1]), title('Mz');
    subplot(2,1,2), plot(reshape(Mxy_highres,1,np*nt)), axis tight, ylim([-1,1]), title('Mxy');
    
    fprintf ( 'Steady-state Mz : %7.5f\n', Mz(np));
    fprintf ( 'Steady-state Mxy: %7.5f\n', Mxy(np));
         
end



       