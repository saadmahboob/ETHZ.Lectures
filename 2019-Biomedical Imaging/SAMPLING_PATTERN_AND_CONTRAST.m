%=========================================================================3
%
%                                                                     
%       BIOMEDICAL IMAGING
%       MRI 2
%
%=========================================================================

%=========================================================================
%	SAMPLING PATTERN, CONTRAST
%=========================================================================

function [] = SAMPLING_PATTERN_AND_CONTRAST()

    clear all; close all; 
    
    [tissue_map,nx,ny,nz,n_tissues] = load_tissue_map('head.bin');          % loads maps of distribution of 12 tissue types in the human head
                                                                            % values indicate relative content (total = 1)
                                                                            % nx = 362, ny = 434, nz = 3 (slices)
                                                                            % data courtesy McGill University, Canada
    
    [rho, T1, T2] = load_tissue_parameters(n_tissues);                      % loads tissue parameters: proton density, T1, T2. Can be manipulated in the function
    
    signal = zeros(nx,ny,nz); 
    
    % Proton density contrast
    for t = 1:n_tissues
        signal(:,:,:) = signal(:,:,:) + rho(t)*tissue_map(:,:,:,t); 
    end        
   
    sampling_pattern = zeros(nx,ny);   
    
    %Full sampling
    sampling_pattern(:,:) = 1;                                               % equal to 1 where k-space is sampled, 0 elsewhere 
          
    for z = 1:nz
        
        figure
        
        subplot(1,3,1), imshow(mat2gray(abs(signal(:,:,z)))), title('Available Signal');       
        
        signal_in_kspace = fftshift(fft2(fftshift(signal(:,:,z))));                         % Signal in k-space where sampling is performed
     
        data = signal_in_kspace(:,:) .* sampling_pattern;                                   % Data obtained by sampling along sampling pattern (using gradient encoding!)  
        
        subplot(1,3,2), imshow(mat2gray(abs(data(:,:)).^0.2)), title('Sampled k-Space');    % raised to the power of 0.2 to visualize small values as well
        
        image = fftshift(ifft2(fftshift(data)));                                            % Image obtained by Fourier Transform  
        
        subplot(1,3,3), imshow(mat2gray(abs(image(:,:)))), title('Image');
        
    end
end

function [rho, T1, T2] = load_tissue_parameters(n_tissues)

    rho = zeros(1,n_tissues);
    T1 = zeros(1,n_tissues);
    T2 = zeros(1,n_tissues);
    
    % Grey Matter
    rho(1)  =   0.9;
    T1(1)   =   920;
    T2(1)   =   100;
    
    % White Matter
    rho(2)  =   0.85;
    T1(2)   =   600;
    T2(2)   =   50;   
    
    % Cerebrospinal Fluid
    rho(3)  =   1.0;
    T1(3)   =   3000;
    T2(3)   =   2000;
    
    % Skull
    rho(4)  =   0.1;
    T1(4)   =   1100;
    T2(4)   =   2;
        
    % Bone Marrow
    rho(5)  =   0.8;
    T1(5)   =   340;
    T2(5)   =   40;
    
    % Fat    
    rho(6)  =   0.8;
    T1(6)   =   240;
    T2(6)   =   40;
    
    % Connective Tissue 
    rho(7)  =   0.5;
    T1(7)   =   240;
    T2(7)   =   70;
    
    % Muscle/Skin
    rho(8)  =   0.8;
    T1(8)   =   880;
    T2(8)   =   100;
    
    % Muscle
    rho(9)  =   0.8;
    T1(9)   =   880;
    T2(9)   =   50;
    
    % Dura Mater
    rho(10)  =   0.3;
    T1(10)   =   800;
    T2(10)   =   20;
    
    % Blood Vessels
    rho(11)  =   1.0;
    T1(11)   =   1350;
    T2(11)   =   200;
    
    % Background
    rho(12)  =   0.0;
    T1(12)   =   1;
    T2(12)   =   1; 
end

function [tissue_map,nx,ny,nz,n_tissues] = load_tissue_map(file_name)
    fileID = fopen(file_name);
    ny = 362;
    nx = 434;
    nz = 3;
    n_tissues = 12;
    tissue_map = zeros(nx,ny,nz,n_tissues);
    map_size = [nx,ny];
    for t = 1:n_tissues
        for z = 1:nz
            tissue_map(:,:,z,t) = fread(fileID,map_size,'*ubit8');
        end
    end
    fclose(fileID);
    for z=1:nz
        figure
        for t = 1:n_tissues
            if t == 1
                name = 'Gray Matter';
            elseif t == 2         
                name = 'White Matter';
            elseif t == 3  
                name = 'CSF';
            elseif t == 4  
                name = 'Skull';
            elseif t == 5  
                name = 'Marrow';
            elseif t == 6  
                name = 'Fat';
            elseif t == 7  
                name = 'Conn. Tissue';
            elseif t == 8  
            	name = 'Muscle/Skin';
            elseif t == 9  
                name = 'Muscle';
            elseif t == 10
                name = 'Dura Mater';
            elseif t == 11  
                name = 'Blood Vessels';
            elseif t == 12  
                name = 'Background';
            end             
            subplot(3,4,t),imshow(mat2gray(tissue_map(:,:,z,t))), title(name);
         end
    end
end


       