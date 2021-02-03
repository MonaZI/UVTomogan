function [rec_img, rec_pdf] = EM_ct_bu(V_init, theta_disc, rdn_mtx_x, rdn_mtx_theta, proj_len, sigma, proj, tilt_series, wedge_sz, I_true, index)
% EM for the tomographic reconstruction problem
% V_init: initial image
% theta_disc: the discretized thetas
% rdn_mtx: the projection matrix

L = size(proj,2);
sz = [length(V_init),1];
N = sqrt(length(V_init));
max_iter = 30;
error = zeros(max_iter,1);

% optimization
% no wedge
%lamb =[4e4, 5e6];
%rho_n = [4e4, 5e6];
% body snr=inf
lamb =[4e0, 1e0]; %1e4
rho_n = [4e1, 5e1]; %5e5
%
% with wedge
%lamb =[4e5, 1e4];
%rho_n = [4e5, 1e4];
%lamb =[4e1, 1e-3];
%rho_n = [4e1, 1e-3];
G = LinOpGrad([N, N]);
tt = LinOpShape([N^2, 1], [N, N]);
G = G*tt;
Reg = CostL1(G.sizeout,zeros(G.sizeout));
%Reg = CostL1([],zeros(sz));
% Reg2 = CostL1([],K);
Hn = {LinOpIdentity(sz), G};
R_pos = CostNonNeg(sz);

%% precomputations
r = ones(L, length(theta_disc))/length(theta_disc);
%for i=1:L
%    r(i, index(i)) = 1;
%end
turn_im = 1;
turn_angle = 0;
for iter=1:max_iter
    % perform E-step
    if turn_angle==1
        if sigma==0
            % when sigma=0, do template matching instead of fuzzy assignment
            tmp = rdn_mtx_x*V_init;
            tmp = reshape(tmp,[proj_len, length(theta_disc)]);        
            tmp_norm = sqrt(sum(tmp.^2, 1));
            tmp = bsxfun(@rdivide, tmp, tmp_norm);
            %sqrt(sum(tmp.^2, 1))
            for i = 1:L
                temp = bsxfun(@times,tmp,proj(:,i));
                temp = sum(temp,1);
                index_min = find(temp==max(temp));
                r(i, index_min) = 1;
            end
        else
            if tilt_series==false
                tmp = rdn_mtx_x*V_init;
                tmp = reshape(tmp,[proj_len, length(theta_disc)]);
                p_theta = sum(r,1)/sum(r(:));
                rec_pdf(:, iter) = p_theta;
                
                for i = 1:L
                    temp = bsxfun(@minus,tmp,proj(:,i));
                    nom = p_theta.*exp(-sum(temp.^2,1)/(2*2*sigma^2));
                    r(i,:) = nom/sum(nom);
                end
            else
                tmp = rdn_mtx_theta*V_init;
                tmp = reshape(tmp,[proj_len*(2*wedge_sz+1), length(theta_disc)]);
                p_theta = sum(r,1)/sum(r(:));
                rec_pdf(:, iter) = p_theta;
                for i = 1:L
                    temp = bsxfun(@minus,tmp,proj(:,i));
                    nom = p_theta.*exp(-sum(temp.^2,1)/(2*(2*wedge_sz+1)*sigma^2));
                    r(i,:) = nom/sum(nom);
                end
            end
        end  
        turn_angle=0;
        turn_im = 1;
    end
    % perform M-step-------------------------------------------------------
    if turn_im==1
    mat = zeros(length(V_init),length(V_init));
    vec = zeros(length(V_init),1);
    p_tmp = sum(r,1);
    
    for theta_ind = 1:length(theta_disc)
        if tilt_series==false
            angle_index = (theta_ind-1) * proj_len + [0:1:proj_len-1];
            angle_index = angle_index(:)+1;
            temp_mat = rdn_mtx_x(angle_index, :);
        else
            angle_index = ((theta_ind-1) + [-wedge_sz:wedge_sz]);
            angle_index = mod(angle_index, length(theta_disc));
            angle_index = bsxfun(@plus, [0:1:proj_len-1].', angle_index * proj_len);
            angle_index = angle_index(:)+1;
            temp_mat = rdn_mtx_x(angle_index, :);
        end

%         temp_mat = rdn_mtx((theta_ind-1)*proj_len+1:theta_ind*proj_len,:);
        tmp = bsxfun(@times,temp_mat.'*proj,r(:,theta_ind).');
        tt = (temp_mat.')*temp_mat;
        mat = mat + p_tmp(theta_ind)*tt;
        vec = vec + sum(tmp,2);
    end
    mat = mat;
    vec = vec;
    max(abs(mat(:)))
    max(abs(vec(:)))
    % construct the optimization env
    
    mtx = LinOpMatrix(mat);
    size(mat)
    norm(mat*V_init-vec)
    
    LS=CostL2([],vec);            % Least-Squares data term
    F=LS*mtx;
    Fn = {lamb(1)*R_pos,lamb(2)*Reg};
    ADMM = OptiADMM(F,Fn,Hn,rho_n);
    %ADMM = OptiConjGrad(mat, vec)
    ADMM.ItUpOut=1;             % call OutputOpti update every ItUpOut iterations
    ADMM.maxiter=6;           % max number of iterations
    ADMM.run(V_init);   % run the algorithm
    
    error(iter) = norm(V_init(:)-ADMM.OutOp.evolxopt{end}(:),'fro');
    fprintf('error = %f \n',error(iter))
    
    V_init = ADMM.OutOp.evolxopt{end};
    rec_img(:,:,iter) = reshape(V_init,[N,N]).';
    save(['./temp/imgs_body_noise_zeros.mat'], 'rec_img')
    
    if iter==24
        for j=1:24
            subplot(3, 8, j); 
            imagesc(rec_img(:,:,j));
            colormap gray
        end
    end   
    turn_angle=1;
    turn_im=0;
    end
end

end
