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
    def train_epoch(self):
        """
        Trains the discriminator and reconstructs the signal
        """
        for epoch in range(self.args.num_epoch):
            # updating the discriminator
            # deactivate the gradient for the image here
            for iter, real_meas in enumerate(self.dataloader):
                real_meas = real_meas.cuda()

                for param in self.x.parameters():
                    param.requires_grad = False

                if not self.args.pdf_known and not self.args.fixed_pdf:
                    self.p = self.p.detach().cpu()
                    self.pdf = self.pdf.detach().cpu()

                for param in self.net.parameters():
                    param.requires_grad = True

                if self.args.pdf_known or self.args.fixed_pdf:
                    if self.args.pdf_known:
                        angle_indices = np.sort(angle_pdf_samples(self.args.pdf_vec, real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                    elif self.args.fixed_pdf:
                        angle_indices = np.sort(angle_pdf_samples(self.pdf.cpu().numpy(), real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                else:
                    _, angle_indices = gumbel_softmax_sampler(self.pdf, real_meas.shape[0], self.args.tau)
                _, syn_meas = self.x.forward(angle_indices)

                # interpolation term
                if self.args.lamb>0:
                    alpha = torch.rand((real_meas.shape[0], 1)).float().cuda()
                    tmp = alpha * real_meas + (1-alpha) * syn_meas
                    int_meas = Variable(tmp.data, requires_grad=True)
                    int_meas = int_meas.cuda()
                    out = self.net(int_meas)
                    gradients = torch.autograd.grad(outputs=out, inputs=int_meas, grad_outputs=torch.ones(out.shape).cuda())[0].squeeze()
                    tt = gradients.norm(2, dim=1)
                    reg = ((tt-1) ** 2).mean()
                else:
                    reg = 0.

                loss_real = torch.mean(self.net(real_meas))
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
                   
                    if not self.args.pdf_known and not self.args.fixed_pdf: 
                        self.p.requires_grad = True
                        self.pdf = self.Softmax(self.p).float().cuda()

                    if self.args.pdf_known or self.args.fixed_pdf:
                        if self.args.pdf_known:
                            angle_indices = np.sort(angle_pdf_samples(self.args.pdf_vec, real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                        elif self.args.fixed_pdf:
                            angle_indices = np.sort(angle_pdf_samples(self.pdf.detach().cpu().numpy(), real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                            
                        _, syn_meas_noisy = self.x.forward(angle_indices)
                        loss_x = -1*torch.mean(self.net(syn_meas_noisy))
                        reg_pdf = 0.
                        tv_pdf = 0.
                    else:
                        angle_probs, _ = gumbel_softmax_sampler(self.pdf, self.args.batch_size, self.args.tau)
                        angle_probs = angle_probs.cuda()
                        angle_indices = torch.arange(0, self.args.angle_disc, 1)
                        syn_meas_clean, syn_meas_noisy = self.x.forward(angle_indices)
                        #if self.args.sigma>0:
                        #   syn_meas_noisy = syn_meas_clean.unsqueeze(0) + self.args.sigma * torch.randn(self.args.batch_size, 1, syn_meas_clean.shape[1]).repeat(1, self.args.angle_disc, 1).float().cuda()
                        #   syn_meas_noisy = syn_meas_noisy.view(self.args.batch_size*self.args.angle_disc, syn_meas_clean.shape[1])

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
                    if not self.args.pdf_known and not self.args.fixed_pdf:
                        self.p.retain_grad()
                    loss_x.backward()
                    torch.nn.utils.clip_grad_norm_(self.x.parameters(), 10)
                    self.optim_x.step()
                    self.scheduler_x.step()
                    
                    # compute the SSIM and MS-SSIM between the orig and updated image
                    snr = SNR(self.x.image.detach().cpu(), self.image_true.cpu())

                    grad_p = 0.
                    if not self.args.pdf_known and not self.args.fixed_pdf:
                        #grad_p = torch.autograd.grad(loss_x, self.p)[0] #+ 1e-5 * self.p
                        grad_p = self.p.grad
                        grad_p = grad_p / torch.norm(grad_p)
                        self.p = self.p - self.args.lrate_pdf * grad_p
                        self.pdf = self.Softmax(self.p)

                    if not self.args.pdf_known:
                        # compute total variation norm between the pdfs
                        total_v_pdf = torch.norm(self.pdf.detach().cpu()-self.args.pdf_vec, p=1)
                        self.logger_tf.add_scalar('total_v_pdf', total_v_pdf, self.iteration)
                    else:
                        total_v_pdf = torch.tensor(0)

                    self.logger_tf.add_scalar('lr-x', self.optim_x.param_groups[0]['lr'], self.iteration)
                    self.logger_tf.add_scalar('loss-x', loss_x.data.cpu().numpy().item(), self.iteration)
                    self.logger_tf.add_scalar('SNR', snr, self.iteration)

                    if self.args.pdf_known or self.args.fixed_pdf:
                        print('epoch=%d, iter=%d, loss_x=%f, loss_net=%f' %(epoch, self.iteration, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item()))
                    else:
                        print('epoch=%d, iter=%d, loss_x=%f, loss_net=%f, grad_p=%f' %(epoch, self.iteration, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item(), grad_p.max().cpu().numpy()))

                if (self.iteration%self.args.iter_log==0) and (self.iteration>0):
                    self.log(grad_p=grad_p)
                    fig , axes = plt.subplots(1, 5, figsize=(20, 4))
                    im = axes[0].imshow(self.image_true.squeeze().cpu().numpy())
                    axes[0].set_title('GT im, iter={0:1.1f}k'.format(self.iteration/1000.), fontsize=8)
                    
                    image = self.x.relu(self.x.image)
                    axes[1].imshow((image).data.squeeze().cpu().numpy())
                    axes[1].set_title('Recon im, snr={0:1.2f} dB'.format(snr), fontsize=8)

                    axes[2].imshow(real_meas.squeeze().cpu().numpy().transpose())
                    axes[2].set_title('Real projs', fontsize=8)

                    axes[3].imshow(syn_meas.detach().squeeze().cpu().numpy().transpose())
                    axes[3].set_title('Syn projs', fontsize=8)
                    
                    axes[4].plot(np.linspace(0, 1*np.pi, self.args.angle_disc), self.args.pdf_vec, label='pdf-gt')

                    if not self.args.pdf_known:
                        aligned_pdf = self.pdf.detach().cpu().numpy()
                        axes[4].plot(np.linspace(0, 1*np.pi, self.args.angle_disc), aligned_pdf, label='pdf-pred')
                        axes[4].set_title('tv={0:1.2f}'.format(total_v_pdf.numpy()), fontsize=8)

                    self.logger_tf.add_figure('projections_'+str(self.iteration), fig, global_step=self.iteration)
                
                # increase the frequency of updating the image after some iterations
                if self.args.pdf_known and self.iteration==30000:
                    self.args.n_disc = 2
                
                if (self.iteration%200000==0):
                    self.args.tau = max(0.5, np.exp(-3e-5*epoch))

                if (self.iteration%(2*self.args.iter_change_lr)==0) and (self.iteration>0.):
                    self.args.lrate_pdf *= self.args.gamma_x
                
                if self.iteration%1000==0:
                    if self.iteration==0:
                        savedict = {}
                    savedict[str(self.iteration)] = {}
                    savedict[str(self.iteration)]['image_gt'] = self.image_true.squeeze().cpu().numpy()
                    savedict[str(self.iteration)]['image_recon'] = self.x.image.detach().cpu().numpy()
                    if not self.args.pdf_known:
                        savedict[str(self.iteration)]['pdf_est'] = self.pdf.detach().cpu().numpy()
                    savedict[str(self.iteration)]['pdf_gt'] = self.args.pdf_vec
                    scipy.io.savemat('./results/results_' + self.args.exp_name + '.mat', savedict, do_compression=True)
                
                self.iteration += 1
