import os
import scipy.io
import yaml
import cv2
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from projection_2d_1d import *
from utils import *

from dataloader import Dataset, get_loader
from model import Net
from trainer_2d import Trainer

from torch import cuda
import torch
import torch.backends.cudnn as cudnn


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_gen', type=str, default='./configs/config_gen.yaml', help='the general configuration')
    parser.add_argument('-config_exp', type=str, default='./configs/config_exp.yaml', help='the exp configuration')
    args = parser.parse_args()
    return args


def main(args):
    random_seed(args.seed)
    if args.debug:
        args.num_meas = 20

    # load the image
    if args.image_file=='phantom':
        image = scipy.io.loadmat('/home/mona/projects/lib/astra-toolbox/samples/python/phantom.mat')['phantom256']
        image = downsample(image, args.dl_scale)
    else:
        image_sz = int(256/args.dl_scale)
        image = cv2.imread(os.path.join('./data/', args.image_file+'.png'), 0)
        image = cv2.resize(image, (image_sz, image_sz)).astype('float')
        image /= 255.

    #pdf = np.random.uniform(size=[args.angle_disc,])
    #pdf /= np.sum(pdf)
    pdf = sig_from_a(np.random.uniform(size=(args.a_size,))-0.5, args.angle_disc)
    pdf -= np.min(pdf)
    pdf += 0.2
    pdf /= np.sum(pdf)

    args.proj_size = image.shape[0]
    image = torch.tensor(image).float().cuda()
    proj_obj = Project2D(image.shape[0], image.shape[1], args.angle_disc, args.proj_size)

    angle_indices = np.sort(angle_pdf_samples(pdf, args.num_meas, args.angle_disc, pdf_type=args.pdf))
    # sample estimation of the PMF
    temp = np.zeros((args.angle_disc, ))
    for k in range(args.angle_disc):
        temp[k] = np.sum((angle_indices==k))/args.num_meas
    pdf = temp
    args.pdf_vec = pdf
    projs_clean = proj_obj.forward(image, angle_indices, is_cuda=False)
    projs_clean = projs_clean.cpu().numpy()
    image /= projs_clean.max()
    projs_clean /= projs_clean.max()
    var_proj = np.var(projs_clean)
    if args.snr=='inf':
        args.sigma = 0
    else:
        args.sigma = np.sqrt(var_proj/args.snr)
        # snr = 10*np.log10(var_proj/(args.sigma**2))
    # add noise to projections
    projs = projs_clean + args.sigma * np.random.normal(size=projs_clean.shape)
    
    # save the required data for EM
    savedict = {}
    savedict['proj_mat'] = proj_obj.W_tensor.numpy()
    savedict['angle_indices'] = angle_indices
    savedict['image'] = image.cpu().numpy()
    savedict['projs_clean'] = projs_clean
    savedict['projs_noisy'] = projs
    savedict['sigma'] = args.sigma
    savedict['pdf'] = pdf
    scipy.io.savemat('./results/'+args.exp_name+'_EM.mat', savedict, do_compression=True)

    args.use_gpu = (args.gpu is not None)
    if args.use_gpu: cuda.set_device(args.gpu)
    dataset = Dataset(projs, args)
    dataloader = get_loader(dataset, args, is_test=False)

    net = Net(args)
    print(net)

    trainer = Trainer(net, dataloader, image.shape[0], args, proj_obj, image)
    trainer.train()


if __name__=='__main__':
    args = arg_parse()
    args = read_yaml(args.config_gen, args.config_exp)
    main(args)
