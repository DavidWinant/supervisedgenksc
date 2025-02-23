import torch.nn as nn
from utils import Lin_View


class Net1(nn.Module):
    """Encoder - network architecture"""

    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net1, self).__init__()  # inheritance used here.
        self.args = args
        self.main = nn.Sequential(
            nn.Conv2d(nChannels, self.args.capacity, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.args.capacity, self.args.capacity * 2, **cnn_kwargs[0]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(self.args.capacity * 2, self.args.capacity * 4, **cnn_kwargs[1]),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(self.args.capacity * 4 * cnn_kwargs[2] ** 2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, self.args.x_fdim2),
            # nn.Sigmoid() #TODO: check behavior of final activation function (phi(xi)^t*phi(xj)>=0 forall i,j)
        )

    def forward(self, x):
        return self.main(x)


class Net3(nn.Module):
    """Decoder - network architecture"""

    def __init__(self, nChannels, args, cnn_kwargs):
        super(Net3, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, self.args.capacity * 4 * cnn_kwargs[2] ** 2),
            nn.LeakyReLU(negative_slope=0.2),
            Lin_View(self.args.capacity * 4, cnn_kwargs[2], cnn_kwargs[2]),  # Unflatten
            nn.ConvTranspose2d(
                self.args.capacity * 4, self.args.capacity * 2, **cnn_kwargs[1]
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(
                self.args.capacity * 2, self.args.capacity, **cnn_kwargs[0]
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(self.args.capacity, nChannels, **cnn_kwargs[0]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)


class SimpleNet1(nn.Module):
    def __init__(self, ipVec_dim, args):
        super(SimpleNet1, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(ipVec_dim, self.args.x_fdim1),
            # nn.BatchNorm1d(self.args.x_fdim1),
            nn.ReLU(),
            nn.Linear(self.args.x_fdim1, self.args.x_fdim2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class SimpleNet2(nn.Module):
    def __init__(self, ipVec_dim, args):
        super(SimpleNet2, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
            nn.ReLU(),
            nn.Linear(self.args.x_fdim1, ipVec_dim),
        )

    def forward(self, x):
        return self.main(x)


class ClusterGAN_Enc(nn.Module):
    """Decoder - network architecture"""

    def __init__(self, nChannels, args):
        super(ClusterGAN_Enc, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            nn.Conv2d(
                nChannels, self.args.capacity, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(
                self.args.capacity,
                self.args.capacity * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, self.args.x_fdim1),
        )

    def forward(self, x):
        return self.main(x)


class ClusterGAN_Dec(nn.Module):
    """Decoder - network architecture"""

    def __init__(self, nChannels, args):
        super(ClusterGAN_Dec, self).__init__()
        self.args = args
        self.main = nn.Sequential(
            # nn.Linear(self.args.x_fdim2, self.args.x_fdim1),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.args.x_fdim1, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
            # batch normalization
            Lin_View(128, 7, 7),  # Unflatten
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.args.capacity),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nChannels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x)
