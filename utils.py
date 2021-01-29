import yaml
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

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
    reg_loss = torch.sum(torch.abs(x[:-1]-x[1:]))
    return reg_loss


def tv_loss(image):
    reg_loss = torch.sum(torch.abs(image[:, :-1] - image[:, 1:])) + torch.sum(torch.abs(image[:-1, :] - image[1:, :]))
    return reg_loss

def angle_pdf_samples(pdf, num_meas, angle_disc, pdf_type='uniform'):
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
    if pdf.is_cuda:
        shifts = shifts.cuda()
        shift_probs = shift_probs.cuda()
    # optimized
    g = -torch.log(-torch.log(torch.rand(size=(num_meas, len(pdf)))))
    if pdf.is_cuda: g = g.cuda()
    shifts = torch.argmax(torch.log(pdf)+g, dim=1).int().squeeze()
    tmp = torch.exp((torch.log(pdf)+g)/tau)/torch.sum(torch.exp((torch.log(pdf)+g)/tau), dim=1).unsqueeze(1)
    shift_probs = tmp
    return shift_probs, shifts


def sig_from_a(a, sig_len):
    """
    Computes the periodic signal from coefficients a
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


def gauss2D_image(xx, yy, res_val, pixel_size, g_std, weight=1.):
    XX, YY = torch.meshgrid([torch.linspace(-res_val//2, res_val//2-1, res_val), torch.linspace(-res_val//2, res_val//2-1, res_val)])
    proj = torch.zeros(XX.shape)
    if xx.is_cuda:
        XX = XX.cuda()
        YY = YY.cuda()
        proj = proj.cuda()
    XX = XX * pixel_size
    YY = YY * pixel_size
    for k in range(len(xx)):
        tmp = torch.exp(-((XX-xx[k])**2+(YY-yy[k])**2)/(2*g_std**2))
        tmp = weight * tmp/(2*np.pi*g_std**2)
        proj += tmp
    return proj


def align_to_ref(sig, sig_ref):
    """
    Aligns the signal to sig_ref
    :param sig: the signal
    :param sig_ref: the reference signal
    :return: the aligned signal and the shift required for alignment of the two
    """
    # align the ref signal and the recovered one
    res = -1*float('inf')
    for s in range(len(sig)):
        tmp = np.concatenate((sig[s:], sig[0:s]), axis=0)
        inner_prod = np.sum(tmp*sig_ref)
        if inner_prod>res:
            index = s
            res = inner_prod
            sig_aligned = np.concatenate((sig[index:], sig[0:index]), axis=0)
    return sig_aligned, index


def read_yaml(config_gen, config_exp):
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
