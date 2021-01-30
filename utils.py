import yaml
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

def downsample(image, dl_factor):
    """
    Downsamples image by the given dl_factor
    :param image: input image to be downsampled
    :param dl_factor: downsample factor
    :return: the downsampled image
    """
    return image[0:image.shape[0]:dl_factor, 0:image.shape[1]:dl_factor]


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


def tv_loss_pdf(x):
    """
    TV loss applied to 1D signal (the PMF of the projection angles)
    :param x: the input
    :return: the TV loss
    """
    reg_loss = torch.sum(torch.abs(x[:-1]-x[1:]))
    return reg_loss


def tv_loss(image):
    """
    TV loss applied to 2D signal (the image)
    :param x: the input
    :return: the TV loss
    """
    reg_loss = torch.sum(torch.abs(image[:, :-1] - image[:, 1:])) + torch.sum(torch.abs(image[:-1, :] - image[1:, :]))
    return reg_loss


def angle_pdf_samples(pdf, num_meas, angle_disc, pdf_type='uniform'):
    """
    Samples num_meas number of angles based on the given pdf
    :param pdf: PMF  of the projection angles
    :param num_meas: Number of angle samples
    :param angle_disc: Number of bins used to discretize the angles
    :param pdf_type: PDF type for the angle distribution, choices: uniform, nonuniform
    :return: samples drawn from the given PDF
    """
    if pdf_type=='uniform':
        return np.random.randint(0, angle_disc, size=(num_meas,))
    indices = torch.zeros((num_meas,))
    cumsum = np.cumsum(pdf)
    for i in range(num_meas):
        s = np.random.rand(1)
        t = int(np.sum((cumsum<s)))
        indices[i] = t
    return indices


def gumbel_softmax_sampler(pdf, num_meas, tau):
    """
    Draws random samples following pdf distribution
    :param pdf: the pdf
    :param num_meas: the number of samples
    :param tau: the temperature factor
    :return: the set of samples and their softmax approximation
    """
    shifts = torch.zeros(size=(num_meas,), dtype=torch.int)
    shift_probs = torch.zeros(size=(num_meas, len(pdf)))
    g = -torch.log(-torch.log(torch.rand(size=(num_meas, len(pdf)))))
    if pdf.is_cuda:
        shifts = shifts.cuda()
        shift_probs = shift_probs.cuda()
        g = g.cuda()
    shifts = torch.argmax(torch.log(pdf)+g, dim=1).int().squeeze()
    tmp = torch.exp((torch.log(pdf)+g)/tau)/torch.sum(torch.exp((torch.log(pdf)+g)/tau), dim=1).unsqueeze(1)
    shift_probs = tmp
    return shift_probs, shifts


def sig_from_a(a, sig_len):
    """
    Generates a periodic signal from coefficients a
    :param a: the coefficients
    :param sig_len: the length of the signal
    :return: the signal (either a torch tensor or a numpy array)
    """
    if torch.is_tensor(a):
        t = torch.arange(0., sig_len)
        x = torch.zeros((sig_len, ))
        for count, ak in enumerate(a):
            x = x + ak * torch.sin((2*np.pi*count*t)/sig_len)
    else:
        t = np.arange(0., sig_len)
        x = np.zeros((sig_len, ))
        for count, ak in enumerate(a):
            x = x + ak * np.sin((2*np.pi*count*t)/sig_len) # we had a 2 coeff before to make it periodic
    return x


def read_yaml(config_gen, config_exp):
    """
    Read yaml file and return as parsed arguments
    :param config_gen: the path to the general config file (type str)
    :param config_exp: the path to the specific experiment config file (type str)
    :return: parsed arguments
    """
    args = AttrDict()
    with open(config_gen, 'r') as cfg:
        args.update(yaml.safe_load(cfg))
    with open(config_exp, 'r') as cfg:
        args.update(yaml.safe_load(cfg))

    return args


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
