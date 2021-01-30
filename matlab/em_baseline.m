clear all
close all

% load the saved data corresponding to the experiment
%load('../results/exp_body1_64_unknown_wedge0_sigma0_EM.mat')
load('../results/exp_body1_64_unknown_wedge0_snr1_EM.mat')
%load('../results/exp_phantom_64_unknown_wedge0_snr1_comment_EM.mat')
%load('../results/exp_phantom_64_unknown_wedge0_sigma0_EM.mat')
wedge_sz = 1;

proj_size = size(projs_clean, 2);

theta_disc = linspace(0, pi, length(pdf));
angle_indices_bu = angle_indices;

% generate the projections once again
angle_index = bsxfun(@plus, angle_indices.' * proj_size, [0:1:proj_size-1]);
angle_index = angle_index.';
angle_index = angle_index(:)+1;
proj_submat = proj_mat(angle_index, :);

image_tmp = image.';
res = proj_submat * image_tmp(:);
res = reshape(res, [proj_size, length(angle_indices)]);
error = norm(res.'-projs_clean);

projs_noisy = projs_noisy.';

recon_vol = fbp_baseline(proj_noisy);

% EM good init baseline
init_vol = imgaussfilt(image, 3);
init_vol = init_vol.';
rec_img = EM_ct(zeros(size(init_vol(:))), theta_disc, proj_mat, proj_submat, proj_size, sigma, projs_noisy, tilt_series, wedge_sz, image_tmp, angle_indices_bu+1);

% EM poor init baseline






