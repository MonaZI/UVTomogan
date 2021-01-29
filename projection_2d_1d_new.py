import torch
import astra
import numpy as np


def convert_sparse_torch(mat):
    """
    Converts a sparse torch tensor to float
    """
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
        """
        Initialization for the projector class
        :param img_sz_x: width of the image
        :param img_sz_y: height of the image
        :param angle_disc: number of bins (discretizations) for the angles
        :param proj_size: the length of the projection line (usually the same as the image size)
        """
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

    def forward(self, image, angle_index, is_cuda=True):
        """
        Applies the projection operator based on the given angle indices
        :param image: the image to be projected
        :param angle_index: angle indices
        :param is_cuda: final results and tensors put on gpu device or not
        :return:
        """
        batch_size = len(angle_index)
        projs = torch.zeros(batch_size, self.proj_size).cuda()
        bs = 2000
        # If batch_size is very large, it is better to break down the generation of the projection lines in batches.
        # This would be more more time and memory efficient.
        if batch_size>bs:
            for b in range(batch_size//bs):
                print(b)
                angle_index_tmp = angle_index[b*bs:(b+1)*bs]
                angle_index_tmp = np.expand_dims((angle_index_tmp), axis=1) * self.proj_size + np.linspace(0, self.proj_size-1, self.proj_size)
                angle_index_tmp = angle_index_tmp.ravel().astype(int)
                W_tmp = self.W_tensor[angle_index_tmp, :]
                if is_cuda:
                    W_tmp = W_tmp.cuda()
                else:
                    image = image.cpu()
                projs_tmp = torch.matmul(W_tmp, image.flatten())
                projs[b*bs:(b+1)*bs, :] = projs_tmp.view(bs, self.proj_size)
        else:
            angle_index = np.expand_dims((angle_index), axis=1) * self.proj_size + np.linspace(0, self.proj_size-1, self.proj_size)
            angle_index = angle_index.ravel().astype(int)
            W_tmp = self.W_tensor[angle_index, :]
            if is_cuda:
                W_tmp = W_tmp.cuda()
            else:
                image = image.cpu()
            projs = torch.matmul(W_tmp, image.flatten())
            projs = projs.view(batch_size, len(projs)//batch_size)
        return projs
