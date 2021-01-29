import os
from abc import abstractmethod

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class ImageClass(nn.Module):
    """
    The image class
    """
    def __init__(self, args, image_sz, proj_obj):
        """
        Initialing the image class
        :param args: the set of experiment arguments
        :param image_sz: the size of the image (int)
        :param proj_obj: the projector object
        """
        super().__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.Softmax = torch.nn.Softmax(dim=0)
        self.relu = nn.ReLU()
        if self.args.init=='zeros':
            init = torch.zeros((image_sz, image_sz))
        elif self.args.init=='rand':
            init = torch.rand((image_sz, image_sz))*0.1

        self.image = nn.Parameter(init, requires_grad=True)
        self.proj_obj = proj_obj

    def forward(self, angles_index):
        image = self.relu(self.image)
        syn_meas_clean = self.proj_obj.forward(image, angles_index).float().cuda()
        syn_meas_noisy = syn_meas_clean + self.args.sigma * torch.randn(syn_meas_clean.shape).float().cuda()
        return syn_meas_clean, syn_meas_noisy


class TrainerAbstract(object):
    """
    Abstract trainer object
    """
    def __init__(self, net, dataloader, image_sz, args, proj_obj, image_true):
        """
        The initialization of the Trainer object
        :param net: the discriminator network
        :param dataloader: the dataloader object
        :param image_sz: size of the image
        :param args: the set of experiment arguments
        :param proj_obj: the projector object
        :param image_true: the ground truth image (used for comparison only)
        """
        self.args = args
        self.Softmax = torch.nn.Softmax(dim=0)
        self.dataloader = dataloader
        self.image_true = image_true
        self.x = ImageClass(args, image_sz, proj_obj)
        if self.args.use_gpu:
            self.net = net.cuda()
            self.x = self.x.cuda()
        self.logger_tf = SummaryWriter(log_dir=os.path.join(self.args.log_path, self.args.exp_name))

        if not self.args.pdf_known:
            # the pdf is the output of a softmax layer (to have non-negative values and sum up to 1)
            self.p = torch.zeros((self.args.angle_disc,))
            self.pdf = self.Softmax(self.p)

        # generate two different optimizers for the variables and the discriminator network weights
        if self.args.optimizer == 'sgd':
            if not self.args.pdf_known:
                self.optim_pdf = optim.SGD([self.p], lr=self.args.lrate_pdf, weight_decay=0.)
            self.optim_x = optim.SGD(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay_x)
            self.optim_net = optim.SGD(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay)

        elif self.args.optimizer == 'adam':
            if not self.args.pdf_known:
                self.optim_pdf = optim.Adam(self.p, lr=self.args.lrate_pdf, weight_decay=0.)
            self.optim_x = optim.Adam(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay_x)
            self.optim_net = optim.Adam(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay)

        elif self.args.optimizer == 'rms':
            if not self.args.pdf_known:
                self.optim_pdf = optim.RMSprop(self.p, lr=self.args.lrate_pdf, weight_decay=0.)
            self.optim_x = optim.RMSprop(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay_x)
            self.optim_net = optim.RMSprop(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay)

        # set-up learning rate schedulers for the image, pmf and discriminator
        if self.args.scheduler_x == 'step':
            self.scheduler_x = torch.optim.lr_scheduler.StepLR(self.optim_x, step_size=self.args.iter_change_lr_x, gamma=self.args.gamma_x)
            if not self.args.pdf_known:
                self.scheduler_pdf = torch.optim.lr_scheduler.StepLR(self.optim_pdf, step_size=2*self.args.iter_change_lr_x, gamma=self.args.gamma_x)
        elif self.args.scheduler_x == 'cosine':
            self.scheduler_x = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_x, self.args.iter_change_lr_x, eta_min=5e-4)
            if not self.args.pdf_known:
                self.scheduler_pdf = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_pdf, 2*self.args.iter_change_lr_x, eta_min=5e-4)

        if self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim_net, step_size=self.args.iter_change_lr, gamma=self.args.gamma)
        elif self.args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_net, self.args.iter_change_lr, eta_min=5e-4)
        self.iteration = 0

    def train(self):
        """
        Trains and saves the trained model
        :param x_true: the gt signal, only used for comparison with the recon signal
        :param p_true: the gt pdf, only used for comparison with the recon pdf
        :return: nothing is returned
        """
        self.train_epoch()
        print('Finished training!')
        torch.save(self.net.state_dict(), os.path.join(self.args.modelSavePath, self.args.expName))
        return self.x.detach().cpu().numpy()

    @abstractmethod
    def train_epoch(self):
        pass

    def log(self):
        """
        Logs the current values and the gradients of the discriminator network and the signal
        :return: Nothing
        """
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            self.logger_tf.add_histogram(tag, value.data.cpu().numpy(), self.iteration)
            if not(value.grad is None):
                self.logger_tf.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.iteration)

        if not(self.x.image.grad is None):
            self.logger_tf.add_histogram('grad_x', self.x.image.grad.data.cpu().numpy(), self.iteration)
        if not(self.p.grad is None):
            self.logger_tf.add_histogram('grad_p', self.p.grad.data.cpu().numpy(), self.iteration)
        self.logger_tf.add_histogram('x_values', self.x.image.data.cpu().numpy(), self.iteration)
