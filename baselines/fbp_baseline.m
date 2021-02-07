function rec_img = fbp_baseline(projs_noisy, proj_mat)
% TV-Regularized reconstruction
% param proj_noisy: the noisy projections
% param proj_mat: the projection matrix
% return recon_img: the reconstructed image

proj_size = size(projs_noisy, 1);
img_init = zeros(proj_size^2, 1);
sz = [length(img_init),1];
N = sqrt(length(img_init));
% weights for noisy experiments
lamb =[8e1, 1e1];
rho_n = [8e2, 1e2];
% weights for phantom no noise
%lamb = [8e1, 1e1];
%rho_n = [8e2, 1e2];
% weighst for no noise lung
%lamb =[4e0, 5e0];
%rho_n = [4e0, 5e0];

G = LinOpGrad([N, N]);
reshape_op = LinOpShape([N^2, 1], [N, N]);
G = G * reshape_op;
Reg = CostL1(G.sizeout,zeros(G.sizeout));
Hn = {LinOpIdentity(sz), G};
R_pos = CostNonNeg(sz);
mtx = LinOpMatrix(proj_mat);
% data fidelity term
LS = CostL2([],projs_noisy(:));
F = LS * mtx;
Fn = {lamb(1) * R_pos, lamb(2) * Reg};
ADMM = OptiADMM(F, Fn, Hn, rho_n);
ADMM.ItUpOut = 1; 
ADMM.maxiter = 30;
ADMM.run(img_init);
recon_img = ADMM.OutOp.evolxopt{end};
rec_img = reshape(recon_img,[proj_size, proj_size]).';

end
