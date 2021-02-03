% % 2D tomography reconstruction
% % clear all
% 
img_sz = 64;
P = phantom('Modified Shepp-Logan', img_sz);
% P = double(imread('../skull.jpg'));
% P = P(:, :, 1);
P = imresize(P,[img_sz img_sz]);
% P = image;

num_disc = 120;
num_meas = 2000;
angles_range = ([0:1:119]/120)*180;
angles = angles_range;(angle_indices+1);

projs = radon(P, angles);
img_sz = 64;

rec = iradon(projs, angles, 'linear', 'Ram-Lak', 1, img_sz);
figure; imagesc(rec); colormap gray

load('~/Desktop/results.mat')
figure;
colormap gray
subplot(3, 1, 1);
imagesc(P);
title('GT')
subplot(3, 1, 2);
imagesc(rec)
ssimval = ssim(rec, P);
title(['FBP, SSIM = ' num2str(ssimval)])
% title(['SNR = ' num2str(20*log10(norm(P(:))/norm(P(:)-rec(:))))])
subplot(3, 1, 3);
imagesc(image)
ssimval = ssim(double(image), P);
title(['Ours, SSIM = ' num2str(ssimval)])
% title(['SNR = ' num2str(20*log10(norm(P(:))/norm(P(:)-aligned_img(:))))])


