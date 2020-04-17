%=========================================================================
%                                                                     
%       BIOMEDICAL IMAGING
%       II SIGNALS AND SYSTEMS
%
%=========================================================================

%=========================================================================
%	EXERCISE 2, Fast Fourier Transform
%=========================================================================

function [] = SISY_EXERCISE2()

    clear all; close all; 
    
    fprintf ( '-----------------------------------------\n' );  
    fprintf ( '2 FAST FOURIER TRANSFORM                 \n' );  
    fprintf ( '-----------------------------------------\n' );  
    
    n = 256;
    m = 16;
    
    rect = zeros(1,n);                                                             % create rectangle function   
    rect(n/2-m/2+1:n/2+m/2) = 1;
    
    
    % a) Straightforward FFT
    
    trafo = fft(rect);                                                             % Fourier transform
    
    figure;
    range1 = [-0.1 1.1];                                                           % define plot ranges
    range2 = [-0.1 1.1]*max(abs(trafo));
    range3 = [-0.3 1.1]*max(abs(trafo));                                                                  
    range4 = [-1.1 1.1]*max(abs(trafo));
    subplot(2,2,1), plot(rect), axis tight, ylim(range1);                          % plot transform
    subplot(2,2,2), plot(abs(trafo)), axis tight, ylim(range2), title('FFT, magnitude');
    subplot(2,2,3), plot(real(trafo)), axis tight, ylim(range4), title('FFT, real');
    subplot(2,2,4), plot(imag(trafo)), axis tight, ylim(range4), title('FFT, imaginary');
    
    % b) Use fftshift to shift origin in the Fourier domain
    
    trafo = fftshift(fft(rect));     
    
    figure;
    subplot(2,2,1), plot(rect), axis tight, ylim(range1);                                                            % plot result
    subplot(2,2,2), plot(abs(trafo)), axis tight, ylim(range2), title('FFT with fftshift, magnitude');
    subplot(2,2,3), plot(real(trafo)), axis tight, ylim(range3), title('FFT with fftshift, real'); 
    subplot(2,2,4), plot(imag(trafo)), axis tight, ylim(range3), title('FFT with fftshift, imaginary');  
         
    % c) Use phase correction to shift origin the original domain
    
    delta_k = 2*pi/n;                                                              % calculate sample spacing in the Fourier domain
    k = [-n/2:n/2-1]*delta_k;                                                      % create vector of sampling positions in the Fourier domain
    x0 = -(n-1)/2;                                                                 % actual position of first input sample  
    trafo = exp(-i.*k.*x0).*fftshift(fft(rect));                                   % include phase correction after FFT
    
    figure;
    subplot(2,2,1), plot(rect), axis tight, ylim(range1);                                                      % plot shifted and phase-corrected transform
    subplot(2,2,2), plot(abs(trafo)), axis tight, ylim(range2), title('with phase correction, magnitude');
    subplot(2,2,3), plot(real(trafo)), axis tight, ylim(range3), title('with phase correction, real'); 
    subplot(2,2,4), plot(imag(trafo)), axis tight, ylim(range3), title('with phase correction, imaginary');  
    
    % d) use fftshift to shift instead
    
    trafo = fftshift(fft(fftshift(rect)));     
    
    figure;
    subplot(2,2,1), plot(rect), axis tight, ylim(range1);                                                            % plot result
    subplot(2,2,2), plot(abs(trafo)), axis tight, ylim(range2), title('fftshift instead, magnitude');
    subplot(2,2,3), plot(real(trafo)), axis tight, ylim(range3), title('fftshift instead, real'); 
    subplot(2,2,4), plot(imag(trafo)), axis tight, ylim(range3), title('fftshift instead, imaginary');  
     
end
       