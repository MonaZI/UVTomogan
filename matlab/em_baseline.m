clear all
close all

% load the saved data corresponding to the experiment
load('/Users/monazehni/PycharmProjects/2DtomoGAN/tomoGAN/results/exp_body1_64_known_wedge0_sigma0_EM.mat')
% load('../results/exp_body1_64_unknown_wedge0_snr1_EM.mat')
%load('../results/exp_phantom_64_unknown_wedge0_snr1_comment_EM.mat')
%load('../results/exp_phantom_64_unknown_wedge0_sigma0_EM.mat')
wedge_sz = 1;
indices = randi(length(angle_indices), 300, 1);
projs_clean = projs_clean(indices, :, :);
projs_noisy = projs_noisy(indices, :, :);
angle_indices = angle_indices(indices);

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

% fbp_recon = fbp_baseline(projs_noisy, proj_submat);
% figure; imagesc(fbp_recon); colormap gray;

% EM good init baseline
% init_vol = imgaussfilt(image, 3);
% init_vol = init_vol.';
% turn_im = 0;
% em_recon_good_init = EM_ct(zeros(size(init_vol(:))), projs_noisy, theta_disc, proj_mat, sigma, turn_im);

% EM poor init baseline
init_vol = rand(size(image)) * 0.1;
turn_im = 1;
em_recon_random_init = EM_ct(zeros(size(init_vol(:))), projs_noisy, theta_disc, proj_mat, sigma, turn_im);







