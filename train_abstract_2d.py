import os
from abc import abstractmethod

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from projection_2d_1d import *
from utils import *


class ImageClass(nn.Module):
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
        self.relu = nn.ReLU()
        if self.args.init=='zeros':
            init = torch.zeros((image_sz, image_sz))
        elif self.args.init=='rand':
            init = torch.rand((image_sz, image_sz)) * 0.1

        self.image = nn.Parameter(init, requires_grad=True)
        self.proj_obj = proj_obj
    def forward(self, angles_index):
        image = self.relu(self.image)
        syn_meas_clean = self.proj_obj.forward(image, angles_index).float().cuda()
        syn_meas_noisy = syn_meas_clean + self.args.sigma * torch.randn(syn_meas_clean.shape).float().cuda()
        return syn_meas_clean, syn_meas_noisy


class TrainerAbstract(object):
    def __init__(self, net, dataloader, image_sz, args, proj_obj, image_true):
        """
        The initialization for the Trainer object
        :param net: the network
        :param meas: the set of real measurements
        :param dataloader: the dataloader to load the original measurements
        :param args: a set of arguments
        """
        self.args = args
        self.dataloader = dataloader
        self.image_true = image_true
        self.x = ImageClass(args, image_sz, proj_obj)
        if self.args.use_gpu:
            self.net = net.cuda()
            self.x = self.x.cuda()
        self.logger_tf = SummaryWriter(log_dir=os.path.join(self.args.log_path, self.args.exp_name))

        if not self.args.pdf_known:
            # definition of pdf
            self.Softmax = torch.nn.Softmax(dim=0)
            self.p = torch.zeros((self.args.angle_disc,))
            self.pdf = self.Softmax(self.p)

        # generate two different optimizers for the variables and the discriminator network weights
        if self.args.optimizer=='sgd':
            self.optim_x = optim.SGD(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay_x)
            self.optim_net = optim.SGD(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay)
        elif self.args.optimizer=='adam':
            self.optim_x = optim.Adam(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay_x)
            self.optim_net = optim.Adam(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay)
        elif self.args.optimizer=='rms':
            self.optim_x = optim.RMSprop(self.x.parameters(), lr=self.args.lrate_x, weight_decay=self.args.wdecay_x)
            self.optim_net = optim.RMSprop(self.net.parameters(), lr=self.args.lrate, weight_decay=self.args.wdecay)
        
        if (not self.args.scheduler_x=='cosine'):
            self.scheduler_x = torch.optim.lr_scheduler.StepLR(self.optim_x, step_size=self.args.iter_change_lr_x, gamma=self.args.gamma_x)
        else:
            self.scheduler_x = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim_x, self.args.iter_change_lr_x, eta_min=5e-4)

        if self.args.scheduler=='step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim_net, step_size=self.args.iter_change_lr, gamma=self.args.gamma)
        elif self.args.scheduler=='cosine':
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
    def train_epoch(self, x_true):
        pass

    def log(self, grad_p=None):
        """
        Logs the current status of the model on val and test splits
        :return: Nothing
        """
        for tag, value in self.net.named_parameters():
            tag = tag.replace('.', '/')
            self.logger_tf.add_histogram(tag, value.data.cpu().numpy(), self.iteration)
            self.logger_tf.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), self.iteration)

        self.logger_tf.add_histogram('grad_x', self.x.image.grad.data.cpu().numpy(), self.iteration)
        self.logger_tf.add_histogram('x_values', self.x.image.data.cpu().numpy(), self.iteration)

        if not self.args.pdf_known and not self.args.fixed_pdf:
            self.logger_tf.add_histogram('grad_p', grad_p.data.cpu().numpy(), self.iteration)
            self.logger_tf.add_histogram('p_values', self.p.data.cpu().numpy(), self.iteration)
