%=========================================================================
%                                                                     
%       BIOMEDICAL IMAGING
%       II SIGNALS AND SYSTEMS
%
%=========================================================================

%=========================================================================
%	EXERCISE 3, Building a Comb
%=========================================================================

function [] = SISY_EXERCISE3()

    clear all; close all; 
    
    fprintf ( '-----------------------------------------\n' );  
    fprintf ( '3 BUILDING A COMB                        \n' );  
    fprintf ( '-----------------------------------------\n' );  
    
    n = 256;                                 % length of inputs
    
    x = [0:n-1]-n/2;
    k = [-n/2:n/2-1]*2*pi/n;                 % sampling positions in the transform domain                                                                                                  
    corr = exp(-i.*k.*x(1));                 % phase correction for shifting origin to the center of the input 
    
    f = zeros(1,n);
    
    
    delta_x = 16
    
    
    for j = 0:n/delta_x/2-1
        f(n/2+1+j*delta_x) = 1
        f(n/2+1-j*delta_x) = 1
        
        g = corr.*fftshift(fft(f))
             
        figure
        subplot(2,2,1), plot(x,f), axis tight, ylim(range(f)), title('rectangle')                                                       
        subplot(2,2,2), plot(k,abs(g)), axis tight, ylim(range(abs(g))), title('FT, magnitude')
        subplot(2,2,3), plot(k,real(g)), axis tight, ylim(range(real(g))), title('FT, real') 
        subplot(2,2,4), plot(k,imag(g)), axis tight, ylim(range(imag(g))), title('FT, imaginary') 
    end        
end

function r = range(x)
    d = 0.1*(max(x)-min(x))
    if (d==0),
        d = 1
    end
    r = [min(x)-d max(x)+d]    
end


       