clear all
close all
rng(0)

% load the saved data corresponding to the experiment
% lung with noise experiment
%load('./data/lung_noise.mat')

% phantom with noise experiment
load('./data/phantom_noisy.mat')

% phantom no noise experiment
%load('./data/phantom_clean.mat')

% lung no noise experiment
%load('./data/lung_clean.mat')

indices = randi(length(angle_indices), size(projs_clean, 1), 1);
%indices = randi(length(angle_indices), 1000, 1); % for debugging 
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

fbp_recon = fbp_baseline(projs_noisy, proj_submat);
save('fbp_recon_phantom_noisy.mat', 'fbp_recon')

% EM good init baseline
init_vol = imgaussfilt(image, 3);
init_vol = init_vol.';
turn_im = 0;
random_init = false;
max_iter = 20;
scale_sigma = 1.;
em_recon_good_init = EM_ct(init_vol(:), projs_noisy, theta_disc, proj_mat, sigma, turn_im, random_init, max_iter, scale_sigma);
save('em_recon_good_init_noisy_phantom.mat', 'em_recon_good_init')

% EM poor init baseline
init_vol = rand(size(image)) * 0.001;
turn_im = 1;
random_init = true;
max_iter = 30;
scale_sigma = 2.;
em_recon_random_init = EM_ct(init_vol(:), projs_noisy, theta_disc, proj_mat, sigma, turn_im, random_init, max_iter, scale_sigma);
save('em_recon_random_init_noisy_phantom.mat', 'em_recon_random_init')
