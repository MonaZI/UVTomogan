% vizualization of the results
close all
clear all

load('../results/results_exp_phantom_64_fixed_wedge1_snr0_2_n60_a3_refined.mat')
image_recon(find(image_recon<0)) = 0;
for i=1:size(image_recon, 3)
    snr_fixed(i) = norm(image_gt-image_recon(:, :, i))/norm(image_gt(:));
    corr_fixed(i) = sum(sum(image_gt.*image_recon(:, :, i)))/norm(image_recon(:, :, i));
end
image_recon_fixed = image_recon;

load('../results/results_exp_phantom_64_unknown_wedge1_snr0_2_n60_a2_refined.mat')
image_recon(find(image_recon<0)) = 0;
for i=1:size(image_recon, 3)
    snr_unknown(i) = norm(image_gt-image_recon(:, :, i))/norm(image_gt(:));
    corr_unknown(i) = sum(sum(image_gt.*image_recon(:, :, i)))/norm(image_recon(:, :, i));
end
image_recon_unknown = image_recon;

load('../results/results_exp_phantom_64_known_wedge0_snr0_0_n120_a2_refined.mat')
image_recon(find(image_recon<0)) = 0;
for i=1:size(image_recon, 3)
    snr_known(i) = norm(image_gt-image_recon(:, :, i))/norm(image_gt(:));
    corr_known(i) = sum(sum(image_gt.*image_recon(:, :, i)))/norm(image_recon(:, :, i));
end
image_recon_known = image_recon;

figure;
plot(corr_known); hold on;
plot(corr_fixed);
plot(corr_unknown)
legend('known', 'fixed', 'unknown')

figure;
subplot(2, 2, 1);
imagesc(image_gt); title('GT'); colormap gray
subplot(2, 2, 2);
imagesc(image_recon_known(:, :, end)); title('Known');
subplot(2, 2, 3);
imagesc(image_recon_fixed(:, :, 50)); title('Fixed with Uniform');
subplot(2, 2, 4);
imagesc(image_recon_unknown(:, :, 100)); title('Unknown');

image_recon(find(image_recon<0)) = 0;

figure;
ind = 1;
for i=1:5:17
    for j=1:2:10
        subplot(4, 5, ind)
        plot(pdf_gt); hold on; plot(pdf_recon(:, (i-1)*10+j))
        title(['iter=' num2str((i-1)*10+j)])
%         imagesc(image_recon_fixed(:, :, (i-1)*17+j))
        ind = ind+1;
    end
end
        
