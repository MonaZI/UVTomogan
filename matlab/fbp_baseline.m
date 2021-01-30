function recon_img = fbp_baseline(proj_noisy)

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
%ADMM.ItUpOut=1;            % call OutputOpti update every ItUpOut iterations
%ADMM.maxiter=30;           % max number of iterations
%ADMM.run(zeros(size(image(:))));   % run the algorithm
%recon_vol = ADMM.OutOp.evolxopt{end};
%rec_img = reshape(recon_vol,[proj_size, proj_size]).';

end
