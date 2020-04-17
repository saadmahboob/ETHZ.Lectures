%=========================================================================
%                                                                     
%       BIOMEDICAL IMAGING
%       II SIGNALS AND SYSTEMS
%
%=========================================================================

%=========================================================================
%	EXERCISE 1, Plane Waves
%=========================================================================

function [] = SISY_EXERCISE1()

    clear all; close all; 
    
    fprintf ( '-----------------------------------------\n' );  
    fprintf ( '1 PLANE WAVES                            \n' );  
    fprintf ( '-----------------------------------------\n' );  
    
    kx = input('kx [1/mm]: ');                                                      % wave numbers [1/mm]
    ky = input('ky [1/mm]: ');
    
    range = 200;                                                                    % spatial range of calculation [mm]
    [x,y] = meshgrid([-range/2:range/2]);                                           % create grid for wave calculation
    
    wave = exp(i.*(kx.*x + ky.*y));                                                 % calculate plane wave
    
    figure;
    subplot(2,2,1), imagesc(real(wave)), title('real part');
    subplot(2,2,2), imagesc(imag(wave)), title('imaginary part');
    subplot(2,2,3), imagesc(abs(wave),[0 1]), title('magnitude');   
    subplot(2,2,4), imagesc(angle(wave)), title('phase');
    
    wave_LSI = 4*del2(wave);                                                        % perform some SI operation. Example chosen: Laplace operator. The factor 4 corrects for scaling in the discrete Matlab implementation.
    
    figure;
    subplot(2,2,1), imagesc(real(wave_LSI)), title('after LSI: real part');
    subplot(2,2,2), imagesc(imag(wave_LSI)), title('after LSI: imaginary part');
    subplot(2,2,3), imagesc(abs(wave_LSI),[0 1]), title('after LSI: magnitude');   
    subplot(2,2,4), imagesc(angle(wave_LSI)), title('after LSI: phase');
    
end
    