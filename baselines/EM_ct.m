function rec_img = EM_ct(V_init, proj, theta_disc, rdn_mtx_x, sigma, turn_im, random_init, max_iter, sigma_scale)
% EM for the tomographic reconstruction problem
% V_init: initial image
% proj: the projection lines
% theta_disc: the discretized projection angles
% rdn_mtx_x: the projection matrix
% sigma: std of the noise
% max_iter: maximum number of iterations
% sigma_scale: the scale of sigma, used for the E-step

proj_len = size(proj, 1);
L = size(proj,2);
sz = [length(V_init),1];
N = sqrt(length(V_init));
error = zeros(max_iter,1);

% optimization
if random_init
    % phantom clean and noisy
    lamb =[4e5, 5e7];
    rho_n = [4e5, 5e7];

    % lung clean and noisy
    %lamb = [4e4, 5e6];
    %rho_n = [4e4, 5e6];
else
    % clean phantom and lung
    %lamb =[4e0, 1e0];
    %rho_n = [4e1, 5e1];

    % noisy lung
    %lamb =[4e4, 1e5]; %1e4
    %rho_n = [4e4, 5e6]; %5e5

    % noisy phantom
    lamb =[4e0, 1e0]; %1e4
    rho_n = [8e0, 5e1]; %5e5
end

G = LinOpGrad([N, N]);
tt = LinOpShape([N^2, 1], [N, N]);
G = G*tt;
Reg = CostL1(G.sizeout,zeros(G.sizeout));
Hn = {LinOpIdentity(sz), G};
R_pos = CostNonNeg(sz);

r = ones(L, length(theta_disc))/length(theta_disc);

turn_angle = ~turn_im;
for iter=1:max_iter
    % perform E-step
    if turn_angle==1
        if sigma==0
            % when sigma=0, do template matching instead of fuzzy assignment
            tmp = rdn_mtx_x*V_init;
            tmp = reshape(tmp,[proj_len, length(theta_disc)]);        
            tmp_norm = sqrt(sum(tmp.^2, 1));
            tmp = bsxfun(@rdivide, tmp, tmp_norm);
            for i = 1:L
                temp = bsxfun(@times,tmp,proj(:,i));
                temp = sum(temp,1);
                index_min = find(temp==max(temp));
                r(i, index_min) = 1;
            end
        else
            tmp = rdn_mtx_x*V_init;
            tmp = reshape(tmp,[proj_len, length(theta_disc)]);
            p_theta = sum(r,1)/sum(r(:));
            rec_pdf(:, iter) = p_theta;
            
            for i = 1:L
                temp = bsxfun(@minus,tmp,proj(:,i));
                % for other noisy experiments 
                nom = p_theta.*exp(-sum(temp.^2,1)/(2*sigma_scale*sigma^2));
                % for random noisy experiment for lung and phantom
                %nom = p_theta.*exp(-sum(temp.^2,1)/(2*2.*sigma^2));
                r(i,:) = nom/sum(nom);
            end
        end  
        turn_angle=0;
        turn_im = 1;
    end
    
    % perform M-step
    if turn_im==1
        mat = zeros(length(V_init),length(V_init));
        vec = zeros(length(V_init),1);
        p_tmp = sum(r,1);
        
        for theta_ind = 1:length(theta_disc)
            angle_index = (theta_ind-1) * proj_len + [0:1:proj_len-1];
            angle_index = angle_index(:)+1;
            temp_mat = rdn_mtx_x(angle_index, :);
            
            tmp = bsxfun(@times,temp_mat.'*proj,r(:,theta_ind).');
            tt = (temp_mat.')*temp_mat;
            mat = mat + p_tmp(theta_ind)*tt;
            vec = vec + sum(tmp,2);
        end
        
        % construct the optimization env
        mtx = LinOpMatrix(mat);
        % norm(mat*V_init - vec)
        
        LS = CostL2([], vec);
        F = LS * mtx;
        Fn = {lamb(1) * R_pos, lamb(2) * Reg};
        ADMM = OptiADMM(F, Fn, Hn, rho_n);
        ADMM.ItUpOut = 1;
        ADMM.maxiter = 6;
        ADMM.run(V_init);
        
        error(iter) = norm(V_init(:)-ADMM.OutOp.evolxopt{end}(:),'fro');
        fprintf('iter=%d/%d, error=%f \n', iter, max_iter, error(iter))
        
        V_init = ADMM.OutOp.evolxopt{end};
        rec_img(:,:,iter) = reshape(V_init,[N,N]).';
        %save(['temp.mat'], 'rec_img')
        
        turn_angle=1;
        turn_im=0;
    end
end

end
