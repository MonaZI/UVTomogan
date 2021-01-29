import time
import torch
import astra
import numpy as np


def downsample(image, dl_factor):
    return image[0:image.shape[0]:dl_factor, 0:image.shape[1]:dl_factor]


def convert_sparse_torch(mat):
    values = mat.data
    coo_data = mat.tocoo()
    indices = torch.LongTensor([coo_data.row,coo_data.col])
    mat_torch = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [mat.shape[0], mat.shape[1]])
    return mat_torch


class Project2D(object):
    """
    Class for projecting 2D signals into 1D signals
    """
    def __init__(self, img_sz_x, img_sz_y, angle_disc, proj_size):
        self.angle_disc = angle_disc
        self.proj_size = proj_size
        angles = np.linspace(0, angle_disc-1, angle_disc)
        angles *= 1*(np.pi/angle_disc)
        # generating the projector matrix
        vol_geom = astra.create_vol_geom(img_sz_x, img_sz_y)
        proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, angles)

        proj_id = astra.create_projector('line', proj_geom, vol_geom)
        matrix_id = astra.projector.matrix(proj_id)
        W = astra.matrix.get(matrix_id)
        self.W = W
        self.W_tensor = convert_sparse_torch(self.W).to_dense()

    def forward(self, image, angle_index, adjoint=False, is_cuda=True, tilt_series=False, wedge_sz=6):
        batch_size = len(angle_index)

        angle_index = np.expand_dims((angle_index), axis=1) * self.proj_size + np.linspace(0, self.proj_size-1, self.proj_size)
        angle_index = angle_index.ravel().astype(int)
        W_tmp = self.W_tensor[angle_index, :]
        if is_cuda:
            W_tmp = W_tmp.cuda()
        else:
            image = image.cpu()
        projs = torch.matmul(W_tmp, image.flatten())
        projs = projs.view(batch_size, len(projs)//batch_size) #.transpose(0, 1)
        return projs
