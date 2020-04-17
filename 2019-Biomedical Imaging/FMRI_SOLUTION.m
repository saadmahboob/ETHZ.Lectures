%=========================================================================
%                                                                     
%       BIOMEDICAL IMAGING
%       MRI 3
%
%=========================================================================

%=========================================================================
%	FMRI
%=========================================================================

function [] = FMRI()

    clear all; close all; 
    
    % Load data
    [nx,ny,nt,fmri_data,anatomical_data,paradigm] = load_data();                % Load data
                                                                                % anatomical image of one brain slice 
                                                                                % series of 200 functional images of the same slice
                                                                                % paradigm = time course of visual stimulus
            
    % Show anatomical data
    figure
    imshow(mat2gray(anatomical_data)), title('Anatomical');
    
    % Show paradigm
    figure
    plot(paradigm),ylim([-2 2]);
    
    % Show functional data  
    figure
    for i = 1:10
        t = 20*(i-1)+1;
        subplot(2,5,i), imshow(mat2gray(fmri_data(:,:,t))), title(strcat('Frame ',num2str(t)));
    end
    
    % TASK: Try to find pixels with temporal fluctuation that resembles the paradigm
    
    figure
    x = 168
    y = 398
    plot(squeeze(fmri_data(y,x,:))), title('Pixel that resembles the paradigm');
    
    
    % TASK: Calculate an activation map
    
    product = zeros(ny,nx,nt);
    for t = 1:nt
        product(:,:,t) = fmri_data(:,:,t)*paradigm(t);
    end
    scalar_product = sum(product,3);
        
    figure
    imagesc(scalar_product), title('Activation map');
    
        
    % TASK: Calculate map of temporal standard deviation, determine noise level
    
    std_dev = std(fmri_data,0,3);
    figure
    imshow(mat2gray(log(std_dev))), title('STD map');
    
    noise_level = sqrt(mean(mean(std_dev(1:50,1:50).^2)))                     % estimate overall standard deviation of noise from area outside the head
         
    
    % TASK: Display thresholded activation map superimposed on anatomical data
    
    mask = abs(scalar_product) > 5*sqrt(nt)*noise_level;                      % threshold activation at a multiple (here 5x) of what is expected from noise only
      
    figure
    imshow(mat2gray(2*anatomical_data.*(1-mask)+mask.*scalar_product)), title('Activation superimposed on anatomy');
         
end



function [nx,ny,nt,fmri_data,anatomical_data,paradigm] = load_data()

    nx = 382;
    ny = 482;
    nt = 200;
    
    fmri_data = zeros(ny,nx,nt);
    anatomical_data = zeros(ny,nx);
    paradigm = zeros(1,nt);
    
    fileID = fopen('fMRI_data.bin');
    for t = 1:nt
        fmri_data(:,:,t) = fread(fileID,[ny,nx],'float');
    end
    fclose(fileID);
    
    fileID = fopen('anatomical_data.bin');
    anatomical_data(:,:) = fread(fileID,[ny,nx],'float');
    fclose(fileID);
    
    fileID = fopen('paradigm.bin');
    paradigm(:) = fread(fileID,[nt],'float');
    fclose(fileID);
    
end
    
