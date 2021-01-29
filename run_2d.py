import sys
sys.path.append("/home/mona/projects/2DtomoGAN")

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


def random_seed(seed):
    """
    Fixes the random seed across all random generators
    :param seed: the seed
    :return: nothing is returned
    """
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    random_seed(args.seed)
    if args.debug:
        args.num_meas = 20

    # load the image
    if args.image_file=='phantom':
        image = scipy.io.loadmat('/home/mona/projects/lib/astra-toolbox/samples/python/phantom.mat')['phantom256']
        image = downsample(image, args.dl_scale)
    elif args.image_file=='head' or args.image_file=='skull' or args.image_file=='body' or args.image_file=='abdomen' or args.image_file=='body1':
        image_sz = int(256/args.dl_scale)
        if args.image_file=='head':
            image = cv2.imread('./head.jpg', 0)
        elif args.image_file=='skull':
            image = cv2.imread('./imgs/skull.jpg', 0)
        elif args.image_file=='body':
            image = cv2.imread('./imgs/body.jpeg', 0)
        elif args.image_file=='abdomen':
            image = cv2.imread('./abdomen.png', 0)
        elif args.image_file=='body1':
            image = cv2.imread('./data/body.png', 0)
        image = cv2.resize(image, (image_sz, image_sz)).astype('float')
        import pdb; pdb.set_trace()
        image /= 255.
    elif args.image_file=='square':
        image = np.zeros(image.shape)
        image[image.shape[0]//4:3*image.shape[0]//4, image.shape[0]//4:3*image.shape[0]//4] = 1.
        image[image.shape[0]//4:3*image.shape[0]//4, :] = 1.
        image[:, image.shape[0]//4:3*image.shape[0]//4] = 1.
    elif args.image_file=='gauss':
        x, y = np.meshgrid(np.linspace(-1,1,image.shape[0]), np.linspace(-1,1,image.shape[1]))
        d = np.sqrt(x*x/2.+y*y)
        sigma, mu = .4, 0.0
        image = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2  )  ) )
    elif args.image_file=='point':
        loc = (np.random.uniform(size=(args.num_point, 2)) - 0.5) * 2.
        radial = np.sqrt(np.sum(loc**2, axis=1))
        res_val = int(2 * np.ceil(1.1*np.max(radial)/args.pixel_size))
        loc = torch.tensor(loc)
        image = gauss2D_image(loc[:, 0], loc[:, 1], res_val, args.pixel_size, args.g_std).cpu().numpy()
        args.scale_pixel = image.max()
        image /= image.max()
        args.res_val = res_val

    savedict = {}
    savedict['image'] = image
    scipy.io.savemat(args.image_file+str(args.dl_scale)+'.mat', savedict)
    args.image = image
    #pdf = np.random.uniform(size=[args.angle_disc,])
    #pdf /= np.sum(pdf)
    pdf = sig_from_a(np.random.uniform(size=(args.a_size,))-0.5, args.angle_disc)
    pdf -= np.min(pdf)
    pdf += 0.2
    pdf /= np.sum(pdf)

    args.pdf_vec = pdf
    args.proj_size = image.shape[0]
    image = torch.tensor(image).float().cuda()
    proj_obj = Project2D(image.shape[0], image.shape[1], args.angle_disc, args.proj_size)

    angle_indices = np.sort(angle_pdf_samples(pdf, args.num_meas, args.angle_disc, pdf_type=args.pdf))
    temp = np.zeros((args.angle_disc, ))
    for k in range(args.angle_disc):
        temp[k] = np.sum((angle_indices==k))/args.num_meas
    pdf = temp
    args.pdf_vec = pdf
    projs_clean = proj_obj.forward(image, angle_indices, is_cuda=False, tilt_series=args.tilt_series, wedge_sz=args.wedge_sz)
    projs_clean = projs_clean.cpu().numpy()
    image /= projs_clean.max()
    projs_clean /= projs_clean.max()
    var_proj = np.var(projs_clean)
    if args.snr==0:
        args.sigma = 0
    else:
        args.sigma = np.sqrt(var_proj/args.snr)
        snr = 10*np.log10(var_proj/(args.sigma**2))
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
    trainer.train(image, projs, image)


if __name__=='__main__':
    args = arg_parse()
    args = read_yaml(args.config_gen, args.config_exp)
    # convert dictionary into attributes
    main(args)
