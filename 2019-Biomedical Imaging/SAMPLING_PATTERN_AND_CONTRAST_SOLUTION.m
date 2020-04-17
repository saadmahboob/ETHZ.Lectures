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

function [] = SAMPLING_PATTERN_AND_CONTRAST_SOLUTION()

    clear all; close all; 
    
    [tissue_map,nx,ny,nz,n_tissues] = load_tissue_map('head.bin');          % loads maps of distribution of 12 tissue types in the human head
                                                                            % values indicate relative content (total = 1)
                                                                            % nx = 362, ny = 434, nz = 3 (slices)
                                                                            % data courtesy McGill University, Canada
    
    [rho, T1, T2] = load_tissue_parameters(n_tissues);                      % loads tissue parameters: proton density, T1, T2. Can be manipulated in the function.
    
    signal = zeros(nx,ny,nz); 
    
    % Contrast
    contrast_choice = 1
    if contrast_choice == 0
        % Proton density contrast
         for t = 1:n_tissues
             signal(:,:,:) = signal(:,:,:) + rho(t)*tissue_map(:,:,:,t); 
         end        
    elseif contrast_choice == 1
        % T2 contrast
        TE = 50;                                                            % echo time [ms]
        angle = 90 * pi/180;                                                 % flip angle [rad]
        for t = 1:n_tissues
            signal(:,:,:) = signal(:,:,:) + sin(angle)*exp(-TE/T2(t))*rho(t)*tissue_map(:,:,:,t); 
        end        
    elseif contrast_choice == 2
        % T1 contrast by inversion recovery
        TI = 700;                                                            % inversion time [ms]
        TE = 2;                                                              % echo time [ms]
        angle = 90 * pi/180;                                                 % flip angle [rad]
        for t = 1:n_tissues
            signal(:,:,:) = signal(:,:,:) + sin(angle)*(1-2*exp(-TI/T1(t)))*exp(-TE/T2(t))*rho(t)*tissue_map(:,:,:,t); 
        end  
    elseif contrast_choice == 3
        % T1 contrast in steady state
        TE = 2;                                                              % echo time [ms]
        TR = 5;                                                              % repetition time [ms]
        angle = 50 * pi/180;                                                 % flip angle [rad]
        for t = 1:n_tissues
            signal(:,:,:) = signal(:,:,:) + sin(angle)*(1-exp(-TR/T1(t)))/(1-cos(angle)*exp(-TR/T1(t)))*exp(-TE/T2(t))*rho(t)*tissue_map(:,:,:,t); 
        end 
    end
      
    sampling_pattern = zeros(nx,ny);                                        % equal to 1 where k-space is sampled, 0 elsewhere
   
    % Sampling extent
    sampling_extent_choice = 0;
    if sampling_extent_choice == 0
        % Full sampling
        sampling_pattern(:,:) = 1;
    elseif sampling_extent_choice == 1
        % Extent reduced to one-half
        sampling_pattern(nx/2-round(nx/4)+1:nx/2+round(nx/4),ny/2-round(ny/4)+1:ny/2+round(ny/4)) = 1;
    elseif sampling_extent_choice == 2
        % Extent reduced to one-fourth
        sampling_pattern(nx/2-round(nx/8)+1:nx/2+round(nx/8),ny/2-round(ny/8)+1:ny/2+round(ny/8)) = 1;
    elseif sampling_extent_choice == 3
        % Extent reduced to one-eighth
        sampling_pattern(nx/2-round(nx/16)+1:nx/2+round(nx/16),ny/2-round(ny/16)+1:ny/2+round(ny/16)) = 1;
    end
    
    %Sampling density
    sampling_density_choice = 0; 
    if sampling_density_choice == 0
        % Full sampling                                                 
    elseif sampling_density_choice == 1
        % vertical undersampling 2x 
        for kx = 1:2:nx sampling_pattern(kx,:) = 0; end                     
    elseif sampling_density_choice == 2
        % vertical undersampling 3x
        for kx = 1:3:nx sampling_pattern(kx,:) = 0; end
        for kx = 3:3:nx sampling_pattern(kx,:) = 0; end   
    elseif sampling_density_choice == 3
        % horizontal undersampling 2x
        for ky = 1:2:ny sampling_pattern(:,ky) = 0; end                     
    elseif sampling_density_choice == 4
        % horizontal undersampling 3x
        for ky = 1:3:ny sampling_pattern(:,ky) = 0; end
        for ky = 3:3:ny sampling_pattern(:,ky) = 0; end                     % start at ky = 2 to hit the center of k-kspace 
    elseif sampling_density_choice == 5
        % 2D undersampling
        for kx = 1:2:nx sampling_pattern(kx,:) = 0; end
        for ky = 1:2:ny sampling_pattern(:,ky) = 0; end
    end
        
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


       