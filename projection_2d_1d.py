import time
import torch
import astra
import numpy as np


def downsample(image, dl_factor):
    return image[0:image.shape[0]:dl_factor, 0:image.shape[1]:dl_factor]


def angle_gen(num_meas, angle_disc):
    angles = np.random.randint(len_disc, size=[num_meas,1])
    angles *= 2*np.pi*(1./len_disc)
    return angles


def sino_gen(image, proj_size, angles):
    proj = Project2D.apply(image, proj_size, angles)
    return proj


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
        #self.W = torch.randn(angle_disc * proj_size, img_sz_x*img_sz_y)
        # convert the sparse np matrix to sparse tensor

    def forward(self, image, angle_index, adjoint=False, is_cuda=True, tilt_series=False, wedge_sz=6):
        batch_size = len(angle_index)
        if tilt_series:
            #start = time.time()
            angle_index_bu = angle_index
            projs = torch.zeros(batch_size, 2*wedge_sz+1, self.proj_size).cuda()
            if batch_size>10000:
                bs = 10000
                for b in range(batch_size//10000):
                    print(b)
                    angle_index_tmp = angle_index[b*bs:(b+1)*bs]
                    tmp = np.linspace(-wedge_sz, wedge_sz, 2*wedge_sz+1)
                    tmp = np.expand_dims(tmp, axis=0) + np.expand_dims(angle_index_tmp, axis=1)
                    tmp = tmp%self.angle_disc
                    tmp = np.expand_dims(tmp, axis=2)*self.proj_size+np.expand_dims(np.expand_dims(np.linspace(0, self.proj_size-1, self.proj_size), axis=0), axis=0)
                    angle_index = np.reshape(tmp, [tmp.shape[0]*tmp.shape[1]*tmp.shape[2], ]).astype(int)
                    W_tmp = self.W_tensor[angle_index, :]
                    if is_cuda:
                        W_tmp = W_tmp.cuda()
                    else:
                        image = image.cpu()
                    projs_tmp = torch.matmul(W_tmp, image.flatten())
                    projs[b*bs:(b+1)*bs, :, :] = projs_tmp.view(bs, 2*wedge_sz+1, self.proj_size)#.permute([2, 1, 0])
            else:
                tmp = np.linspace(-wedge_sz, wedge_sz, 2*wedge_sz+1)
                tmp = np.expand_dims(tmp, axis=0) + np.expand_dims(angle_index, axis=1)
                tmp = tmp%self.angle_disc
                tmp = np.expand_dims(tmp, axis=2)*self.proj_size+np.expand_dims(np.expand_dims(np.linspace(0, self.proj_size-1, self.proj_size), axis=0), axis=0)
                angle_index = np.reshape(tmp, [tmp.shape[0]*tmp.shape[1]*tmp.shape[2], ]).astype(int)
                #W_tmp = self.W[angle_index, :]
                #start = time.time()
                #W_tmp = convert_sparse_torch(W_tmp)
                #if is_cuda:
                #    W_tmp = W_tmp.cuda()
                #    W_tmp1 = W_tmp1.cuda()
                #else:
                #    image = image.cpu()
                #projs1 = torch.matmul(W_tmp.to_dense(), image.flatten())
                #print(time.time()-start)
                #start = time.time()
                W_tmp = self.W_tensor[angle_index, :]
                if is_cuda:
                    W_tmp = W_tmp.cuda()
                else:
                    image = image.cpu()
                projs = torch.matmul(W_tmp, image.flatten())
                #print(time.time()-start)
                projs = projs.view(batch_size, 2*wedge_sz+1, self.proj_size)#.permute([2, 1, 0])
                #projs2 = projs2.view(self.proj_size, 2*wedge_sz+1, batch_size).permute([2, 1, 0])
                #for count, n in enumerate(angle_index):
                #    angle_wedge = np.linspace(n-wedge_sz, n+wedge_sz, 2*wedge_sz+1)
                #    angle_wedge = angle_wedge%self.angle_disc
                #    angle_index = np.expand_dims((angle_wedge), axis=1) * self.proj_size + np.linspace(0, self.proj_size-1, self.proj_size)
                #    angle_index = angle_index.ravel().astype(int)
                #    #W_tmp = self.W[angle_index, :].squeeze().cuda()
                #    W_tmp = self.W[angle_index, :]
                #    W_tmp = convert_sparse_torch(W_tmp)
                #    if is_cuda:
                #        W_tmp = W_tmp.cuda()
                #    else:
                #        image = image.cpu()
                #    tmp = torch.matmul(W_tmp, image.flatten())
                #    projs[count, :, :] = tmp.view(2*wedge_sz+1, self.proj_size)
        else:
            angle_index = np.expand_dims((angle_index), axis=1) * self.proj_size + np.linspace(0, self.proj_size-1, self.proj_size)
            angle_index = angle_index.ravel().astype(int)
            #W_tmp = self.W[angle_index, :].squeeze().cuda()
            #start = time.time()
            #W_tmp = self.W[angle_index, :]
            #W_tmp = convert_sparse_torch(W_tmp)
            #if is_cuda:
            #    W_tmp = W_tmp.cuda()
            #else:
            #    image = image.cpu()
            #projs = torch.matmul(W_tmp, image.flatten())
            #print(time.time()-start)
            #start = time.time()
            W_tmp = self.W_tensor[angle_index, :]
            if is_cuda:
                W_tmp = W_tmp.cuda()
            else:
                image = image.cpu()
            projs = torch.matmul(W_tmp, image.flatten())
            #print(time.time()-start)
            #import pdb; pdb.set_trace()
            projs = projs.view(batch_size, len(projs)//batch_size) #.transpose(0, 1)
        if adjoint:
            adj = torch.matmul(W_tmp.transpose(0, 1), projs)
            adj = adj.view(image.shape[0], image.shape[1])
            projs = projs.view(batch_size, len(projs)//batch_size) #.transpose(0, 1)
            return projs, adj
        # make sure you are doing the reshaping right
        # I think this is right now, visualize it to make sure about this!
        return projs

    #@staticmethod
    #def forward(ctx, input, proj_size, angles):
    #    # astra.set_gpu_index([input.device.index], memory=1*1024*1024*1024)

    #    # input = input.permute(2, 1, 0).contiguous()
    #    # setup volume
    #    #import pdb; pdb.set_trace()
    #    #x,y = input.shape

    #    #strideBytes = input.stride(-2) * 32/8 # 32 bits in a float, 8 bits in a byte
    #    #vol_link = astra.data.GPULink(input.data_ptr(), x, y, strideBytes)
    #    #vol_geom = astra.create_vol_geom(x,y,z)
    #    #vol_id = astra.data3d.link('-vol', vol_geom, vol_link)

    #    ## setup projection
    #    #proj = torch.empty(projSize, len(vectors), projSize, dtype=torch.float, device='cuda')
    #    #x,y,z = proj.shape
    #    #strideBytes = proj.stride(-2) * 32/8 # 32 bits in a float, 8 bits in a byte
    #    #proj_link = astra.data3d.GPULink(proj.data_ptr(), x, y, z, strideBytes)
    #    #proj_geom = astra.create_proj_geom('parallel3d_vec', projSize, projSize, vectors.numpy())
    #    #proj_id = astra.data3d.link('-sino', proj_geom, proj_link)

    #    # setup volume
    #    x, y = input.shape

    #    vol_geom = astra.create_vol_geom(x, y)
    #    proj_geom = astra.create_proj_geom('parallel', 1.0, proj_size, angles)

    #    # As before, create a sinogram from a phantom
    #    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    #    sinogram_id, sinogram = astra.create_sino(input, proj_id)

    #    # Create a data object for the reconstruction
    #    rec_id = astra.data2d.create('-vol', vol_geom)

    #    # Set up the parameters for a reconstruction algorithm using the GPU
    #    cfg = astra.astra_dict('SIRT_CUDA')
    #    # cfg['ImageDataId'] = vol_id
    #    cfg['ReconstructionDataId'] = rec_id
    #    cfg['ProjectionDataId'] = sinogram_id

    #    # Available algorithms:
    #    # SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA (see the FBP sample)


    #    # Create the algorithm object from the configuration structure
    #    alg_id = astra.algorithm.create(cfg)
    #    astra.algorithm.run(alg_id)

    #    ctx.proj_geom = proj_geom
    #    ctx.vol_geom = vol_geom

    #    return torch.from_numpy(sinogram)

    #@staticmethod
    #def backward(ctx, grad_out):
    #    astra.set_gpu_index([grad_out.device.index], memory=10*1024*1024*1024)

    #    grad_input = gradProjSize = gradVectors = None

    #    if ctx.needs_input_grad[0]:
    #        # permute here to handle opposite data layouts
    #        bproj_id, bproj_data = astra.create_backprojection( grad_out.squeeze(1).permute(2, 0, 1).data.cpu().numpy(), ctx.proj_id)
    #        grad_input = Tensor(bproj_data).cuda(non_blocking=True)

    #    return grad_input, gradProjSize, gradVectors

