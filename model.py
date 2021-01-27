import torch
import torch.nn as nn


class Net(nn.Module):
    """
    The network class (discriminator network)
    """
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.net_gen()

    def net_gen(self):
        """
        Generates the NN
        :return: Nothing is returned
        """
        in_size = self.args.proj_size
        mid_size = self.args.mid_size
        modules_tail = [nn.Linear(in_size, 2 * mid_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(2 * mid_size, mid_size, bias=True),
                    nn.ReLU(),
                    nn.Linear(mid_size, mid_size//2, bias=True),
                    nn.ReLU(),
                    nn.Linear(mid_size//2, mid_size//4, bias=True),
                    nn.ReLU(),
                    nn.Linear(mid_size//4, 1, bias=True)]
        self.tail = nn.Sequential(*modules_tail)

        # initializing the layers
        self.tail.apply(self.weights_init)

    def forward(self, x):
        x = self.tail(x)
        return x

    def weights_init(self, m):
        """
        Initializes the weights of the network
        :param m: the parameters of the network
        :return: Nothing returned, parameters initialized in-place
        """
        classname = m.__class__.__name__
        if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
            m.weight.data.normal_(0.0, 0.05)
            m.bias.data.fill_(0)

