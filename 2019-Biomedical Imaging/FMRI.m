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
    
    % ...
    
    
    % TASK: Calculate an activation map
    
    % ...
    
        
    % TASK: Calculate map of temporal standard deviation, determine noise level
    
    % ...
         
    
    % TASK: Display thresholded activation map superimposed on anatomical data
    
    % ...
             
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
    
