%=========================================================================
%                                                                     
%	TITLE: 
%       XCT - EXERCISE 1
%								
%	DESCRIPTION:						
%       COMPUTE PROJECTIONS OF ANALYTICAL PHANTOM 
%
%	INPUT:								
%       NONE	
%
%	OUTPUT:							
%       DISPLAY
%			
%	VERSION HISTORY:						
%	    150816SK INITIAL VERSION
%	    191020SK UPDATE
%
%=========================================================================

%=========================================================================
%	M A I N  F U N C T I O N
%=========================================================================
function [] = XCT_EXERCISE1()

    clear all; close all; 
    
    
    % --------------------------------------------------------------------
    % Display title
    % -------------------------------------------------------------------- 
    fprintf ( '-----------------------------------------\n' );  
    fprintf ( ' BIOMEDICAL IMAGING - XCT-EXERCISE #1\n' );  
    fprintf ( '-----------------------------------------\n' );  
    
    
    % --------------------------------------------------------------------
    % Set imaging parameters
    % --------------------------------------------------------------------
    matrix          = 256;                  % image matrix  [1pixel = 1cm]
    ua              = 50;                   % anode voltage [keV]
      
    % --------------------------------------------------------------------
    % TASK 1.1 (begin)
    % --------------------------------------------------------------------
    % Set tissue density
    % (see: Table 2 in www.nist.gov/pml/data/xraycoef/)
    % --------------------------------------------------------------------
    rho_blood       = 1.060;                % density blood    [g/cm3]
    rho_bone        = 1.920;                % density bone     [g/cm3]
    rho_lung        = 0.001;                % density lung     [g/cm3]
    rho_muscle      = 1.050;                % density muscle   [g/cm3]
    
     
    % --------------------------------------------------------------------
    % Set X-ray mass attenuation coefficients for 50 and 150 keV
    % (see: Table 4 in www.nist.gov/pml/data/xraycoef/)
    % --------------------------------------------------------------------
    mac_blood(1)    = 0.228;                % blood @  50 keV   [cm2/g]
    mac_blood(2)    = 0.149;                % blood @ 150 keV   [cm2/g]
    
    mac_bone(1)     = 0.424;                % bone @  50 keV    [cm2/g]
    mac_bone(2)     = 0.148;                % bone @ 150 keV    [cm2/g]
    
    mac_lung(1)     = 0.208;                % lung @  50 keV    [cm2/g]
    mac_lung(2)     = 0.136;                % lung @ 150 keV    [cm2/g]
    
    mac_muscle(1)   = 0.226;                % muscle @  50 keV  [cm2/g]
    mac_muscle(2)   = 0.149;                % muscle @ 150 keV  [cm2/g]
    
    
    % --------------------------------------------------------------------
    % Calculate linear attenuation coefficients
    % --------------------------------------------------------------------
    mue_blood       = rho_blood.*mac_blood(:);    
    mue_bone        = rho_bone.*mac_bone(:); 
    mue_lung        = rho_lung.*mac_lung(:); 
    mue_muscle      = rho_muscle.*mac_muscle(:); 
     
    idx = 1; if ua==150 idx = 2; end        % set ua dependent index
    
   
    % --------------------------------------------------------------------
    % Define analytical phantom using ellipses with [x0 y0 a b phi mue]
    %
    %       x0,y0   - center point [cm] (+x -> left-right, +y -> bottom-up)
    %       a,b     - half axes [cm]
    %       theta   - rotation angle relative to x-axis [deg]
    %       mue     - linear attenuation coefficient [1/cm]
    % --------------------------------------------------------------------
    phantom.ellipse = [-50 0 20 20 0 mue_muscle(idx)];      % test object
    
    
    % --------------------------------------------------------------------
    % Compute and display discrete phantom 
    % --------------------------------------------------------------------
    [x,y] = meshgrid(-fix(matrix/2):+fix(matrix/2));
   
    phantom.discrete = CalcDiscretePhantom(x,y,phantom,ua);
  
    DisplayData(phantom.discrete,[1,4,1]); title('Discrete phantom'); 
    
    
    % --------------------------------------------------------------------
    % Display profile of mue(x)
    % --------------------------------------------------------------------
    DisplayData(phantom.discrete(fix(matrix/2),:)',[1,4,2]); 
    title('Phantom (profile)'); xlabel('x'); ylabel('mue(x)');
    
    
    % --------------------------------------------------------------------
    % Display column-wise sum of mue(x)
    % --------------------------------------------------------------------
    DisplayData(sum(phantom.discrete(:,:),1)',[1,4,3]); 
    title('Phantom (projection)'); xlabel('x'); ylabel('mue(x)');
    
      
    % --------------------------------------------------------------------
    % Compute and display projection
    % --------------------------------------------------------------------
    [r,phi] = meshgrid(-fix(matrix/2):+fix(matrix/2),[0]);  
        
    phantom.projection = CalcLineIntegrals(r,phi,phantom,ua);
    
    DisplayData(phantom.projection',[1,4,4]); 
    title('Projection'); xlabel('r'); ylabel('P(r))'); 
    
    waitforbuttonpress;                                     % pause here
   
    
    % --------------------------------------------------------------------
    % TASK 1.3 (begin) 
    % --------------------------------------------------------------------
    % Define analytical phantom using ellipses with [x0 y0 a b phi mue]
    %
    %       x0,y0   - center point [cm] (+x -> left-right, +y -> bottom-up)
    %       a,b     - half axes [cm]
    %       theta   - rotation angle relative to x-axis [deg]
    %       mue     - linear attenuation coefficient [1/cm]
    % --------------------------------------------------------------------
    phantom.ellipse = [   0   0   90  80  0   mue_muscle(idx);                  % thorax
                          0   0   70  60  0   mue_lung(idx)-mue_muscle(idx);    % lung
                       +110   0   15  15  0   mue_muscle(idx);                  % left arm muscle
                       +110   0    5   5  0   mue_bone(idx)-mue_muscle(idx);    % left arm bone
                       -110   0   15  15  0   mue_muscle(idx);                  % right arm muscle
                       -110   0    5   5  0   mue_bone(idx)-mue_muscle(idx);    % right arm bone
                          0   0   10  10  0   mue_blood(idx)-mue_lung(idx);     % aorta 
                        +30 +25   25  20 35   mue_muscle(idx)-mue_lung(idx)];   % heart
    
     
    % --------------------------------------------------------------------
    % Compute phantom, projection and display
    % --------------------------------------------------------------------
    [x,y]   = meshgrid(-fix(matrix/2):+fix(matrix/2));
    [r,phi] = meshgrid(-fix(matrix/2):+fix(matrix/2),[0]);  
   
    phantom.discrete    = CalcDiscretePhantom(x,y,phantom,ua);
    phantom.projection  = CalcLineIntegrals(r,phi,phantom,ua);

    phantom_no_contrast = phantom.discrete;
    project_no_contrast = phantom.projection;

    DisplayData(phantom.discrete,[2,4,1]); 
    title('Phantom (w/o contrast agent)'); 
                    
    DisplayData(phantom.projection',[2,4,2]); 
    title('Projection (w/o contrast agent)'); xlabel('r'); ylabel('P(r))'); 
   
    
    % --------------------------------------------------------------------
    % Recompute phantom and projections (with contrast agent)
    % --------------------------------------------------------------------
    phantom.ellipse(7,6) = 2*mue_blood(idx);   % aorta with contrast agent
    
    phantom.discrete    = CalcDiscretePhantom(x,y,phantom,ua);
    phantom.projection  = CalcLineIntegrals(r,phi,phantom,ua);
    
    DisplayData(phantom.discrete,[2,4,3]); 
    title('Phantom (with contrast agent)'); 
                    
    DisplayData(phantom.projection',[2,4,4]); 
    title('Projection (with contrast agent)'); xlabel('r'); ylabel('P(r))');
   
    
    % --------------------------------------------------------------------
    % Subtract images, projections and display (with contrast agent)
    % --------------------------------------------------------------------
    phantom_dsa     = phantom.discrete-phantom_no_contrast;
    project_dsa     = phantom.projection-project_no_contrast;
   
    DisplayData(phantom_dsa,[2,4,7]); 
    title('Phantom (DSA)'); 
                    
    DisplayData(project_dsa',[2,4,8]); 
    title('Projection (DSA)'); xlabel('r'); ylabel('P(r))');   
     
end


%=========================================================================
function [image] = CalcDiscretePhantom(x,y,phantom,ua)
    
    image = zeros(size(x));
    
    for k = 1:length(phantom.ellipse(:,1))
        
        theta   = phantom.ellipse(k,5)*pi/180;
        
        X0      = [x(:)'-phantom.ellipse(k,1);y(:)'-phantom.ellipse(k,2)];
        D       = [1/phantom.ellipse(k,3) 0;0 1/phantom.ellipse(k,4)];
        Q       = [cos(theta) sin(theta); -sin(theta) cos(theta)];
         
        % ----------------------------------------------------------------
        % Find inside of ellipse given by X0,D,Q
        % ----------------------------------------------------------------
        equ = sum((D*Q*X0).^2);
        i = find(equ <= 1);
         
        % ----------------------------------------------------------------
        % Assign linear attenuation coefficients
        % ----------------------------------------------------------------
        image(i) = image(i)+phantom.ellipse(k,6);
        
    end
end


%=========================================================================
function [projection] = CalcLineIntegrals(r,phi,phantom,ua)
    
    projection  = zeros(size(r));
    phi         = phi/180*pi;
    
    sinphi  = sin(phi(:)); 
    cosphi  = cos(phi(:));
    
    rx      = r(:).*cosphi; 
    ry      = r(:).*sinphi;
    
    for k=1:length(phantom.ellipse(:,1))
        
        x0      = phantom.ellipse(k,1); y0 = phantom.ellipse(k,2);
        a       = phantom.ellipse(k,3); b  = phantom.ellipse(k,4);
        
        theta   = phantom.ellipse(k,5)*pi/180; 
        mue     = phantom.ellipse(k,6);
        
        r0      = [rx-x0,ry-y0]';
        
        DQ      = [cos(theta)/a sin(theta)/a; -sin(theta)/b cos(theta)/b];
        DQphi   = DQ*[sinphi,-cosphi]'; 
        DQr0    = DQ*r0;
        
        A       = sum(DQphi.^2); 
        B       = 2*sum(DQphi.*DQr0);
        C       = sum(DQr0.^2)-1; 
        equ     = B.^2-4*A.*C;
        
        i       = find(equ>0);
        
        sp      = 0.5*(-B(i)+sqrt(equ(i)))./A(i);
        sq      = 0.5*(-B(i)-sqrt(equ(i)))./A(i);
        
        % ----------------------------------------------------------------
        % TASK 1.2 (begin) 
        % ----------------------------------------------------------------
        proj    = mue*abs(sp-sq); 
        
        % ----------------------------------------------------------------
        % TASK 1.2 (end) 
        % ----------------------------------------------------------------
        
        projection(i) = projection(i)+proj;
       
    end
    
    projection = reshape(projection,size(r));
end


%=========================================================================
%=========================================================================