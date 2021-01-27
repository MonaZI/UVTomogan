import os
import time

import cv2
import numpy as np
import matplotlib
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from train_abstract_2d import TrainerAbstract
from utils import *
from projection_2d_1d import *
from train_abstract_2d import Proteiner
from dataloader import Dataset, get_loader
from model import Net

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
            for iter, real_meas in enumerate(self.dataloader):
                real_meas = real_meas.cuda()

                # freeze the signal and the PMF when updating the discriminator
                for param in self.x.parameters():
                    param.requires_grad = False
                if not self.args.pdf_known:
                    self.p = self.p.detach().cpu()
                    self.pdf = self.pdf.detach().cpu()

                for param in self.net.parameters():
                    param.requires_grad = True

                # TODO: I am here!!!!
                if self.args.pdf_known:
                    if not self.args.pdf_known:
                        angle_indices = np.sort(angle_pdf_samples(self.args.pdf_vec, real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                else:
                    _, angle_indices = gumbel_softmax_sampler(self.pdf, real_meas.shape[0], self.args.tau)
                _, syn_meas = self.x.forward(angle_indices)

                # gradient penalty term
                if self.args.lamb>0:
                    alpha = torch.rand((real_meas.shape[0], 1, 1)).float().cuda()
                    tmp = alpha * real_meas + (1-alpha) * syn_meas
                    int_meas = Variable(tmp.data, requires_grad=True)
                    int_meas = int_meas.cuda()
                    out = self.net(int_meas, self.iteration)
                    gradients = torch.autograd.grad(outputs=out, inputs=int_meas, grad_outputs=torch.ones(out.shape).cuda())[0].squeeze()
                    tt = gradients.norm(2, dim=1)
                    reg = ((tt-1) ** 2).mean()
                else:
                    reg = 0.

                loss_real = torch.mean(self.net(real_meas, self.iteration))
                loss_syn = torch.mean(self.net(syn_meas, self.iteration))
                
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
                self.logger_tf.add_scalar('loss-real', loss_real.data.cpu().numpy().item(), self.iteration)
                self.logger_tf.add_scalar('loss-syn', loss_syn.data.cpu().numpy().item(), self.iteration)

                # updating the signal
                if (self.iteration%self.args.n_disc==0) and (self.iteration>0):
                    # freeze the discriminator when updating the image and the pmf
                    for param in self.net.parameters():
                        param.requires_grad = False

                    for param in self.x.parameters():
                        param.requires_grad = True
                   
                    if not self.args.pdf_known:
                        for param in self.p.parameters():
                            param.requires_grad = True
                        self.pdf = self.Softmax(self.p).float().cuda()

                    if self.args.pdf_known:
                        angle_indices = np.sort(angle_pdf_samples(self.args.pdf_vec, real_meas.shape[0], self.args.angle_disc, pdf_type=self.args.pdf))
                        _, syn_meas_noisy = self.x.forward(angle_indices)
                        # added this
                        syn_meas_noisy = syn_meas_noisy.unsqueeze(1)
                        loss_x = -1*torch.mean(self.net(syn_meas_noisy, self.iteration))
                        reg_pdf = 0.
                        tv_pdf = 0.
                    else:
                        angle_probs, _ = gumbel_softmax_sampler(self.pdf, self.args.batch_size, self.args.tau)
                        angle_probs = angle_probs.cuda()
                        angle_indices = torch.arange(0, self.args.angle_disc, self.args.angle_disc)
                        syn_meas_clean, syn_meas_noisy = self.x.forward(angle_indices)

                        out = self.net(syn_meas_noisy, self.iteration).squeeze()
                        loss_x = -1. * torch.mean(angle_probs * out)
                        reg_pdf = self.args.wdecay_pdf * torch.mean(self.pdf**2)
                        tv_pdf = self.args.tv_pdf * tv_loss_pdf(self.pdf)
                    
                    reg_x = self.args.tv_weight * tv_loss(self.x.image)
                    print('TV=%f, tv_pdf=%f, max_protein=%f, min_protein=%f' %(reg_x, tv_pdf, self.x.image.max(), self.x.image.min()))
                    loss_x += (reg_x + reg_pdf + tv_pdf)
                    self.optim_x.zero_grad()
                    if not self.args.pdf_known:
                        self.optim_pdf.zero_grad()
                    loss_x.backward()
                    torch.nn.utils.clip_grad_norm_(self.x.parameters(), 10)
                    torch.nn.utils.clip_grad_norm_(self.p.parameters(), 1)
                    self.optim_x.step()
                    self.scheduler_x.step()

                    if not self.args.pdf_known:
                        self.optim_pdf.step()
                        self.scheduler_pdf.step()

                        total_v_pdf = torch.norm(self.pdf.detach().cpu()-self.args.pdf_vec, p=1)
                        self.logger_tf.add_scalar('total_v_pdf', total_v_pdf, self.iteration)

                    self.logger_tf.add_scalar('lr-x', self.optim_x.param_groups[0]['lr'], self.iteration)
                    self.logger_tf.add_scalar('loss-x', loss_x.data.cpu().numpy().item(), self.iteration)


                    if self.args.pdf_known or self.args.fixed_pdf: # or ms!=1:
                        print('epoch=%d, iter=%d, loss_x=%f, loss_net=%f, loss_real=%f, loss_syn=%f' %(epoch, self.iteration, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item(), loss_real.detach().cpu().numpy().item(), loss_syn.detach().cpu().numpy().item()))
                    else:
                        print('epoch=%d, iter=%d, loss_x=%f, loss_net==%f' %(epoch, self.iteration, loss_x.detach().cpu().numpy().item(), loss.detach().cpu().numpy().item()))

                if (self.iteration%self.args.iter_log==0) and (self.iteration>0):
                    self.log()
                    if not self.args.pdf_known:
                        fig , axes = plt.subplots(2, 3)
                    else:
                        fig , axes = plt.subplots(2, 2)
                    im = axes[0, 0].imshow(self.x_true.squeeze().cpu().numpy())
                    image = self.x.relu(self.x.image)

                    snr = SNR(image.detach().cpu(), self.x_true.cpu())
                    axes[0, 1].imshow((image).data.squeeze().cpu().numpy())
                    axes[0, 1].set_title('iter={0:1.1f}, snr={1:2.2f}'.format(self.iteration, snr))
                    axes[1, 0].imshow(real_meas.squeeze().cpu().numpy().transpose())
                    axes[1, 1].imshow(syn_meas.detach().squeeze().cpu().numpy().transpose())
                    plt.title('epoch=%d' %epoch)
                    if not self.args.pdf_known:
                        axes[0, 2].plot(np.linspace(0, 1*np.pi, self.args.angle_disc_orig), self.args.pdf_vec, label='pdf-gt')
                        aligned_pdf = self.pdf.detach().cpu().numpy()
                        axes[0, 2].plot(np.linspace(0, 1*np.pi, self.args.angle_disc), aligned_pdf, label='pdf-pred')
                        axes[0, 2].set_title('tv={0:1.2f}'.format(total_v_pdf.numpy()))

                    self.logger_tf.add_figure('projections_'+str(self.iteration), fig, global_step=self.iteration)

                if self.args.pdf_known and self.iteration==30000:
                    self.args.n_disc = 2

                # save intermediate results every 1k iterations
                if self.iteration%1000 == 0:
                    if self.iteration == 0:
                        savedict = {}
                    savedict[str(self.iteration)] = {}
                    savedict[str(self.iteration)]['image_gt'] = self.x_true.squeeze().cpu().numpy()
                    savedict[str(self.iteration)]['image_recon'] = self.x.image.detach().cpu().numpy()
                    if not self.args.pdf_known:
                        savedict[str(self.iteration)]['pdf_est'] = self.pdf.detach().cpu().numpy()
                    savedict[str(self.iteration)]['pdf_gt'] = self.args.pdf_vec
                    scipy.io.savemat('./results/results_' + self.args.exp_name + '.mat', savedict, do_compression=True)
                
                self.iteration += 1
