clear all
close all

% load the data corresponding to the experiment
%load('../results/exp_body1_64_unknown_wedge0_sigma0_EM.mat')
load('../results/exp_body1_64_unknown_wedge0_snr1_EM.mat')
%load('../results/exp_phantom_64_unknown_wedge0_snr1_comment_EM.mat')
%load('../results/exp_phantom_64_unknown_wedge0_sigma0_EM.mat')
wedge_sz = 1;
indices = randperm(20000);
indices = indices(1:20000);
%indices = randi(length(angle_indices), 300, 1);
projs_clean = projs_clean(indices, :, :);
projs_noisy = projs_noisy(indices, :, :);
angle_indices = angle_indices(indices);

tilt_series = false;
if tilt_series==true
    wedge_sz = floor(size(projs_clean, 2)/2);
    proj_size = size(projs_clean, 3);
else
    proj_size = size(projs_clean, 2);
end

theta_disc = linspace(0, pi, length(pdf));
angle_indices_bu = angle_indices;
if tilt_series==false
    % generate the projections once again
    angle_index = bsxfun(@plus, angle_indices.' * proj_size, [0:1:proj_size-1]);
    angle_index = angle_index.';
    angle_index = angle_index(:)+1;
    proj_submat = proj_mat(angle_index, :);
    
    image_tmp = image.';
    res = proj_submat * image_tmp(:);
    res = reshape(res, [proj_size, length(angle_indices)]);
    error = norm(res.'-projs_clean);
else
    %TODO: fix the one with tilt-series
    angle_indices = linspace(0, length(theta_disc)-1, length(theta_disc));
    angle_index = bsxfun(@plus, angle_indices.', [-wedge_sz:wedge_sz]);
    angle_index = mod(angle_index, length(theta_disc));
    tmp = zeros(1, 1, proj_size);
    tmp(1, 1, :) = [0:1:proj_size-1];
    angle_index = repmat(angle_index, [1, 1, proj_size])*proj_size + repmat(tmp, [length(angle_indices), 2*wedge_sz+1, 1]);
    angle_index = permute(angle_index, [3, 2, 1]);
    angle_index = angle_index(:)+1;
    proj_submat = proj_mat(angle_index, :);
%     proj_mat = proj_submat;
    
    image_tmp = image.';
    res = proj_submat * image_tmp(:);
    res = reshape(res, [proj_size * (2*wedge_sz+1), length(angle_indices)]);
%     res = permute(res, [3, 2, 1]);
%     error = norm(res(:)-projs_clean(:));
    
end
if tilt_series==false
    projs_noisy = projs_noisy.';
else
    projs_noisy = permute(projs_noisy, [3, 2, 1]);
    projs_noisy = reshape(projs_noisy, [proj_size*(2*wedge_sz+1), length(angle_indices_bu)]);
end

% TV reg
%V_init = zeros(proj_size^2, 1);
%sz = [length(V_init),1];
%N = sqrt(length(V_init));
%% weights for body noise
%%lamb =[8e1, 1e0];
%%rho_n = [8e2, 1e1];
%% weights for phantom no noise
%lamb = [8e1, 1e1];
%rho_n = [8e2, 1e2];
%% no noise
%%lamb =[4e0, 5e0];
%%rho_n = [4e0, 5e0];
%
%G = LinOpGrad([N, N]);
%tt = LinOpShape([N^2, 1], [N, N]);
%G = G*tt;
%Reg = CostL1(G.sizeout,zeros(G.sizeout));
%Hn = {LinOpIdentity(sz), G};
%R_pos = CostNonNeg(sz);
%mtx = LinOpMatrix(proj_submat);
%projs_clean = projs_clean.';
%LS=CostL2([],projs_noisy(:));            % Least-Squares data term
%F=LS*mtx;
%Fn = {lamb(1)*R_pos,lamb(2)*Reg};
%ADMM = OptiADMM(F,Fn,Hn,rho_n);
%ADMM.ItUpOut=1;             % call OutputOpti update every ItUpOut iterations
%ADMM.maxiter=30;           % max number of iterations
%ADMM.run(zeros(size(image(:))));   % run the algorithm
%recon_vol = ADMM.OutOp.evolxopt{end};
%init_vol = ADMM.OutOp.evolxopt{end};
%rec_img = reshape(recon_vol,[proj_size, proj_size]).';
%init_vol = reshape(init_vol,[proj_size, proj_size]).';
init_vol = imgaussfilt(image, 3);
%%save('./temp/init_phantom.mat', 'init_vol')
init_vol = init_vol.';
rec_img = EM_ct(zeros(size(init_vol(:))), theta_disc, proj_mat, proj_submat, proj_size, sigma, projs_noisy, tilt_series, wedge_sz, image_tmp, angle_indices_bu+1);






