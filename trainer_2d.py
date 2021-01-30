import os
import time

import numpy as np
import matplotlib
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from train_abstract_2d import TrainerAbstract
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


class Trainer(TrainerAbstract):
    def train_epoch(self, x_true, proj_true, adj):
        """
        Trains the discriminator and reconstructs the signal
        :param x_true: the gt signal, only used for comparison with the recon signal
        :param p_true: the gt pdf, only used for comparison with the recon pdf
        :return: nothing is returned
        """
        self.args.angle_disc_orig = self.args.angle_disc
        self.args.lrate_pdf_bu = self.args.lrate_pdf
        self.args.lrate_x_bu = self.args.lrate_x
        for epoch in range(self.args.num_epoch):
            # updating the discriminator
            # deactivate the gradient for the signal here
            for iter, real_meas in enumerate(self.dataloader):
                real_meas = real_meas.cuda()

                for param in self.x.parameters():
                    param.requires_grad = False
                if not self.args.pdf_known:
                    self.p = self.p.detach().cpu()
                    self.pdf = self.pdf.detach().cpu()

                for param in self.net.parameters():
                    param.requires_grad = True

                # interpolation term
                #angle_indices = np.random.randint(0, self.args.angle_disc, size=(real_meas.shape[0],))
                if self.args.pdf_known:
                    angle_indices = np.sort(angle_pdf_samples(self.args.pdf_vec, real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                else:
                    _, angle_indices = gumbel_softmax_sampler(self.pdf, real_meas.shape[0], self.args.tau)
                _, syn_meas = self.x.forward(angle_indices)
                if self.args.with_conv or self.args.tilt_series: syn_meas = syn_meas.unsqueeze(1)

                if self.args.lamb>0:
                    alpha = torch.rand((real_meas.shape[0], 1, 1, 1)).float().cuda()
                    tmp = alpha * real_meas + (1-alpha) * syn_meas
                    int_meas = Variable(tmp.data, requires_grad=True)
                    int_meas = int_meas.cuda()
                    out = self.net(int_meas)
                    # some notes: when you are taking the gradient, make sure that the output is a scalar
                    # both input and output to autograd should have the requires_grad set to True
                    gradients = torch.autograd.grad(outputs=out, inputs=int_meas, grad_outputs=torch.ones(out.shape).cuda())[0].squeeze()
                    tt = gradients.norm(2, dim=1)
                    reg = ((tt-1) ** 2).mean()
                    #print(reg)
                else:
                    reg = 0.

                # real measurements term
                loss_real = torch.mean(self.net(real_meas))

                # measurements from the reconstructed signal term
                loss_syn = torch.mean(self.net(syn_meas))
                
                loss_tmp = -1*(loss_real-loss_syn)
                loss = -1*(loss_real - loss_syn - self.args.lamb * reg)
                self.optim_net.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optim_net.step()
                self.scheduler.step()
                self.logger_tf.add_scalar('lr-disc', self.optim_net.param_groups[0]['lr'], self.iteration)
                self.logger_tf.add_scalar('loss-disc', loss.data.cpu().numpy().item(), self.iteration)
                self.logger_tf.add_scalar('loss-disc-only', loss_tmp.data.cpu().numpy().item(), self.iteration)

                # updating the signal
                if (self.iteration%self.args.n_disc==0) and self.iteration>0:
                    for param in self.net.parameters():
                        param.requires_grad = False

                    for param in self.x.parameters():
                        param.requires_grad = True
                   
                    if not self.args.pdf_known:
                        self.p.requires_grad = True
                        self.pdf = self.Softmax(self.p).float().cuda()

                    if self.args.pdf_known:
                        angle_indices = np.sort(angle_pdf_samples(self.args.pdf_vec, real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                        _, syn_meas_noisy = self.x.forward(angle_indices)
                        if self.args.with_conv or self.args.tilt_series: syn_meas_noisy = syn_meas_noisy.unsqueeze(1)
                        loss_x = -1*torch.mean(self.net(syn_meas_noisy))
                        reg_pdf = 0.
                        tv_pdf = 0.
                    else:
                        angle_probs, _ = gumbel_softmax_sampler(self.pdf, self.args.batch_size, self.args.tau)
                        angle_probs = angle_probs.cuda()
                        angle_indices = torch.arange(0, self.args.angle_disc_orig, self.args.angle_disc_orig/self.args.angle_disc)
                        syn_meas_clean, syn_meas_noisy = self.x.forward(angle_indices)
                        #if self.args.sigma>0:
                        #    if self.args.tilt_series:
                        #        syn_meas_noisy = syn_meas_clean.unsqueeze(0) + self.args.sigma * torch.randn(self.args.batch_size, 1, syn_meas_clean.shape[1], syn_meas_clean.shape[2]).repeat(1, self.args.angle_disc, 1, 1).float().cuda()
                        #        syn_meas_noisy = syn_meas_noisy.view(self.args.batch_size*self.args.angle_disc, syn_meas_clean.shape[1], syn_meas_clean.shape[2])
                        #    else:
                        #        syn_meas_noisy = syn_meas_clean.unsqueeze(0) + self.args.sigma * torch.randn(self.args.batch_size, 1, syn_meas_clean.shape[1]).repeat(1, self.args.angle_disc, 1).float().cuda()
                        #        syn_meas_noisy = syn_meas_noisy.view(self.args.batch_size*self.args.angle_disc, syn_meas_clean.shape[1])

                        if self.args.with_conv or self.args.tilt_series: syn_meas_noisy = syn_meas_noisy.unsqueeze(1)
                        out = self.net(syn_meas_noisy).squeeze()
                        #if self.args.sigma>0:
                        #    out = out.view(self.args.batch_size, self.args.angle_disc)
                        loss_x = -1 * torch.mean(angle_probs*out)
                        reg_pdf = self.args.wdecay_pdf * torch.mean(self.pdf**2)
                        tv_pdf = self.args.tv_pdf * tv_loss_pdf(self.pdf)
                    
                    reg_x = self.args.tv_weight * tv_loss(self.x.image)
                    print('TV=%f, tv_pdf=%f, max_protein=%f, min_protein=%f' %(reg_x, tv_pdf, self.x.image.max(), self.x.image.min()))
                    loss_x += (reg_x + reg_pdf + tv_pdf)
                    self.optim_x.zero_grad()
                    #self.p.register_hook(print)
                    self.p.retain_grad()
                    loss_x.backward()
                    #if self.iteration%300==0: import pdb; pdb.set_trace()
                    torch.nn.utils.clip_grad_norm_(self.x.parameters(), 10)
                    self.optim_x.step()
                    self.scheduler_x.step()
                    
                    # compute the SSIM and MS-SSIM between the orig and updated image
                    ssim_val = ssim(self.x.relu(self.x.image).unsqueeze(0).unsqueeze(1).detach().cpu(), x_true.squeeze().unsqueeze(0).unsqueeze(1).cpu(), data_range=1, size_average=True)
                    #ms_ssim_val = ms_ssim((self.x.protein).unsqueeze(0).unsqueeze(1).detach().cpu(), x_true.squeeze().unsqueeze(0).unsqueeze(1).cpu(), data_range=1, size_average=True)
                    
                    grad_p = 0.
                    if not self.args.pdf_known and not self.args.fixed_pdf:
                        #grad_p = torch.autograd.grad(loss_x, self.p)[0] #+ 1e-5 * self.p
                        grad_p = self.p.grad
                        grad_p = grad_p / torch.norm(grad_p)
                        # grad_p = self.clip_grad(grad_p)
                        self.p = self.p - self.args.lrate_pdf * grad_p
                        self.pdf = self.Softmax(self.p)
                        # compute total variation norm between the pdfs
                        total_v_pdf = torch.norm(self.pdf.detach().cpu()-self.args.pdf_vec, p=1)
                        self.logger_tf.add_scalar('total_v_pdf', total_v_pdf, self.iteration)
                    else:
                        total_v_pdf = torch.tensor(0)

                    self.logger_tf.add_scalar('lr-x', self.optim_x.param_groups[0]['lr'], self.iteration)
                    self.logger_tf.add_scalar('loss-x', loss_x.data.cpu().numpy().item(), self.iteration)
                    self.logger_tf.add_scalar('ssim', ssim_val.item(), self.iteration)
                    #self.logger_tf.add_scalar('ms-ssim', ms_ssim_val.item(), self.iteration)


                    if self.args.pdf_known or self.args.fixed_pdf:
                        print('epoch=%d, iter=%d, loss_x=%f, loss_net=%f' %(epoch, self.iteration, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item()))
                    else:
                        print('epoch=%d, iter=%d, loss_x=%f, loss_net=%f, grad_p=%f' %(epoch, self.iteration, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item(), grad_p.max().cpu().numpy()))

                if (self.iteration%self.args.iter_log==0)  and (self.iteration>0):
                    self.log(grad_p=grad_p)
                    if not self.args.pdf_known:
                        fig , axes = plt.subplots(2, 3)
                    else:
                        fig , axes = plt.subplots(2, 2)
                    im = axes[0, 0].imshow(x_true.squeeze().cpu().numpy())
                    if self.args.image_file=='point':
                        points = (self.x.sigmoid(self.x.image)-0.5)*2.
                        image = gauss2D_image(points[:, 0], points[:, 1], self.args.res_val, self.args.pixel_size, self.args.g_std)
                    else:
                        image = self.x.relu(self.x.image) #self.x.relu(self.x.protein)
                    axes[0, 1].imshow((image).data.squeeze().cpu().numpy())
                    axes[0, 1].set_title('ssim={0:1.2f}'.format(ssim_val.numpy()))
                    if self.args.tilt_series:
                        real_img = np.concatenate([real_meas[i, :, :].squeeze().cpu().numpy() for i in range(6)], axis=0)
                        fake_img = np.concatenate([syn_meas_noisy[i, :, :].detach().squeeze().cpu().numpy() for i in range(6)], axis=0)
                        axes[1, 0].imshow(real_img)
                        axes[1, 1].imshow(fake_img)
                    else:
                        axes[1, 0].imshow(real_meas.squeeze().cpu().numpy().transpose())
                        axes[1, 1].imshow(syn_meas.detach().squeeze().cpu().numpy().transpose())
                    plt.title('epoch=%d' %epoch)
                    if not self.args.pdf_known:
                        axes[0, 2].plot(np.linspace(0, 1*np.pi, self.args.angle_disc_orig), self.args.pdf_vec, label='pdf-gt')
                        aligned_pdf = self.pdf.detach().cpu().numpy()
                        #aligned_pdf, _ = align_to_ref(self.pdf.detach().cpu().numpy(), self.args.pdf_vec)
                        axes[0, 2].plot(np.linspace(0, 1*np.pi, self.args.angle_disc), aligned_pdf, label='pdf-pred')
                        axes[0, 2].set_title('tv={0:1.2f}'.format(total_v_pdf.numpy()))

                    self.logger_tf.add_figure('projections_'+str(self.iteration), fig, global_step=self.iteration)

                if self.args.pdf_known and self.iteration==30000:
                    self.args.n_disc = 2
                
                if (self.iteration%200000==0):
                    self.args.tau = max(0.5, np.exp(-3e-5*epoch))

                if self.iteration<self.args.start_pdf:
                    self.args.lrate_pdf = 0.
                elif self.iteration==self.args.start_pdf:
                    self.args.lrate_pdf = self.args.lrate_pdf_init

                if (self.iteration%(2*self.args.iter_change_lr)==0) and (self.iteration>self.args.start_pdf):
                    self.args.lrate_pdf *= self.args.gamma_x
                
                #if self.iteration==85000:
                #    self.args.fixed_pdf = True
                #    self.args.pdf_known = True
                #    self.args.pdf_vec = self.pdf.detach().cpu().numpy() 
                    
                if self.iteration%1000==0:
                    if self.iteration==0:
                        savedict = {}
                    savedict[str(self.iteration)] = {}
                    savedict[str(self.iteration)]['image_gt'] = x_true.squeeze().cpu().numpy()
                    savedict[str(self.iteration)]['image_recon'] = self.x.image.detach().cpu().numpy()
                    if not self.args.pdf_known:
                        savedict[str(self.iteration)]['pdf_est'] = self.pdf.detach().cpu().numpy()
                    savedict[str(self.iteration)]['pdf_gt'] = self.args.pdf_vec
                    scipy.io.savemat('./results/results_' + self.args.exp_name + '.mat', savedict, do_compression=True)
                
                self.iteration += 1
