import torch
import imageio
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from skimage import img_as_ubyte

# from tensorflow.keras.applications.inception_v3 import preprocess_input
# import cv2
import glob
import os
import skimage.transform
import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from scipy.linalg import sqrtm
import scipy.misc
from skimage.transform import resize

rcParams["animation.convert_path"] = r"/usr/bin/convert"
rcParams["animation.ffmpeg_path"] = r"/usr/bin/ffmpeg"


class create_dirs:
    """Creates directories for logging, Checkpoints and saving trained models"""

    def __init__(self, name, ct):
        self.name = name
        self.ct = ct

        self.dircp = "checkpoint.pth_{}.tar".format(self.ct)
        self.dirout = "{}_Trained_rkm_{}.tar".format(self.name, self.ct)

    def create(self):
        if not os.path.exists("cp/{}".format(self.name)):
            os.makedirs("cp/{}".format(self.name))

        if not os.path.exists("log/{}".format(self.name)):
            os.makedirs("log/{}".format(self.name))

        if not os.path.exists("out/{}".format(self.name)):
            os.makedirs("out/{}".format(self.name))

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, "cp/{}/{}".format(self.name, self.dircp))

    def save_cluster_checkpoint(self, state, k):
        dircluster = "{}_clusters.pth_{}.tar".format(k, self.ct)
        torch.save(state, "cp/{}/{}".format(self.name, dircluster))


def convert_to_imshow_format(image):
    # Convert from CHW to HWC
    if image.shape[0] == 1:
        return image[0, :, :]
    else:
        if np.any(np.where(image < 0)):
            # First convert back to [0,1] range from [-1,1] range
            image = image / 2 + 0.5
        return image.transpose(1, 2, 0)


def _get_traversal_range(max_traversal, mean=0, std=1):
    """Return the corresponding traversal range in absolute terms."""

    if max_traversal < 0.5:
        max_traversal = (1 - 2 * max_traversal) / 2  # from 0.45 to 0.05
        max_traversal = stats.norm.ppf(
            max_traversal, loc=mean, scale=std
        )  # from 0.05 to -1.645

    # Symmetrical traversals
    return (-1 * max_traversal, max_traversal)


class Lin_View(nn.Module):
    """Unflatten linear layer to be used in Convolution layer"""

    def __init__(self, c, a, b):
        super(Lin_View, self).__init__()
        self.c, self.a, self.b = c, a, b

    def forward(self, x):
        try:
            return x.view(x.size(0), self.c, self.a, self.b)
        except:
            return x.view(1, self.c, self.a, self.b)


class Resize:
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float32 array
        return skimage.util.img_as_float32(resize_image)


# def scatter_w_hist(h):
#     """ 2D scatter plot of latent variables"""
#     fig = plt.figure()
#     grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
#     main_ax = fig.add_subplot(grid[:-1, 1:])
#     y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
#     x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
#
#     main_ax.scatter(h[:, 0].detach().numpy(), h[:, 1].detach().numpy(), s=1)
#
#     _, binsx, _ = x_hist.hist(h[:, 0].detach().numpy(), 40, histtype='stepfilled', density=True,
#                               orientation='vertical')
#     _, binsy, _ = y_hist.hist(h[:, 1].detach().numpy(), 40, histtype='stepfilled', density=True,
#                               orientation='horizontal')
#     x_hist.invert_yaxis()
#     y_hist.invert_xaxis()
#     plt.setp(main_ax.get_xticklabels(), visible=False)
#     plt.setp(main_ax.get_yticklabels(), visible=False)
#     plt.show()


def scatter_w_labels(h, d1, d2, labels, var):
    """2D scatter plot of latent variables"""
    # Make the figures large enough

    h_norm = torch.norm(h, dim=1, keepdim=True)
    max_h_norm = torch.max(h_norm)

    fig = plt.figure()
    grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    # y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    # x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    scatter = main_ax.scatter(
        h[:, d1].detach().numpy(),
        h[:, d2].detach().numpy(),
        s=1,
        c=labels,
        cmap="tab10",
    )

    main_ax.legend(*scatter.legend_elements())

    # _, binsx, _ = x_hist.hist(h[:, d1].detach().numpy(), 40, histtype='stepfilled', density=True,
    #                           orientation='vertical')
    # _, binsy, _ = y_hist.hist(h[:, d2].detach().numpy(), 40, histtype='stepfilled', density=True,
    #                           orientation='horizontal')
    # x_hist.invert_yaxis()
    # y_hist.invert_xaxis()
    plt.setp(main_ax.get_xticklabels(), visible=False)
    plt.setp(main_ax.get_yticklabels(), visible=False)

    if var == "h":
        # plt.title('Latent Space h{} - h{} with 3 Clusters'.format(d1+1,d2+1),fontsize=30)
        plt.title("Latent Space with 3 Clusters", fontsize=30)
    elif var == "e":
        plt.title("Score variables e{} - e{}".format(d1 + 1, d2 + 1))

    plt.show()

    # """ 2D scatter plot of latent variables"""
    # fig = plt.figure()
    # grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
    # main_ax = fig.add_subplot(grid[:-1, 1:])
    # y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    # x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    #
    # main_ax.scatter(h[:, 0].detach().numpy(), h[:, 1].detach().numpy(), s=1)
    # _, binsx, _ = x_hist.hist(h[:, 0].detach().numpy(), 40, histtype='stepfilled', density=True,
    #                           orientation='vertical')
    # _, binsy, _ = y_hist.hist(h[:, 1].detach().numpy(), 40, histtype='stepfilled', density=True,
    #                           orientation='horizontal')
    # x_hist.invert_yaxis()
    # y_hist.invert_xaxis()
    # plt.setp(main_ax.get_xticklabels(), visible=False)
    # plt.setp(main_ax.get_yticklabels(), visible=False)
    # plt.show()


def scatter3_w_labels(h, d1, d2, d3, labels, k, var):
    """3D scatter plot of latent variables"""

    h_norm = torch.norm(h, dim=1, keepdim=True)
    max_h_norm = torch.max(h_norm)

    fig = plt.figure()
    # grid = plt.GridSpec(4, 4, hspace=0.0, wspace=0.0)
    main_ax = fig.add_subplot(111, projection="3d")
    # y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
    # x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
    scatter = main_ax.scatter(
        h[:, d1].detach().numpy(),
        h[:, d2].detach().numpy(),
        h[:, d3].detach().numpy(),
        s=5,
        c=labels,
        cmap="tab10",
    )
    # if k == 2:
    #     prototypes = np.array([[1, 0, 0], [-1, 0, 0]])*max_h_norm
    # elif k == 3:
    #    prototypes = simplex_coordinates1(k-1)*max_h_norm
    #    prototypes = np.hstack((prototypes, np.zeros((prototypes.shape[0], 1))))
    # else:
    #     prototypes = (simplex_coordinates1(k-1)*10**(-5)).transpose()
    # main_ax.scatter(prototypes[:, d1], prototypes[:, d2], prototypes[:, d3], s=100, c='black', marker='x')
    main_ax.legend(*scatter.legend_elements())

    # _, binsx, _ = x_hist.hist(h[:, d1].detach().numpy(), 40, histtype='stepfilled', density=True,
    #                           orientation='vertical')
    # _, binsy, _ = y_hist.hist(h[:, d2].detach().numpy(), 40, histtype='stepfilled', density=True,
    #                           orientation='horizontal')
    # x_hist.invert_yaxis()
    # y_hist.invert_xaxis()
    # plt.setp(main_ax.get_xticklabels(), visible=False)
    # plt.setp(main_ax.get_yticklabels(), visible=False)

    if var == "h":
        # plt.title('Latent Space h{} - h{} - h{} with {} clusters'.format(d1+1,d2+1, d3 +1, k),fontsize=30)
        plt.title("Latent Space with {} clusters".format(k), fontsize=30)
    elif var == "e":
        plt.title("Score variables e{} - e{}-e{}".format(d1 + 1, d2 + 1, d3 + 1))

    plt.show()


class fid:
    """Calculates the Frechet Inception Distance"""

    def __init__(self, gt, gen):
        from tensorflow.keras.applications.inception_v3 import InceptionV3

        self.gt = gt
        self.gen = gen
        self.model = InceptionV3(
            include_top=False, pooling="avg", input_shape=(299, 299, 3)
        )  # prepare the inception v3 model

    # Scale an array of images to a new size
    def scale_images(self, images, new_shape):
        images_list = list()
        for image in images:
            # Resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # Store
            images_list.append(new_image)
        return asarray(images_list)

    # Calculate frechet inception distance
    def calculate_fid(self, model, images1, images2):
        # Calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # Calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)

        # Calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)

        # Calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))

        # Check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real

        # Calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def comp_fid(self):
        fid = np.empty(10)
        im_vae1 = []
        files = glob.glob("{}/*.png".format(self.gt))  # your image path
        for myFile in files:
            image = cv2.imread(myFile)
            im_vae1.append(image)

        im_vae1 = np.array(im_vae1, dtype="float32")
        images1 = im_vae1.astype("float32")
        images1 = self.scale_images(images1, (299, 299, 3))
        images1 = preprocess_input(images1)

        for i in range(10):
            im_vae2 = []
            files2 = glob.glob("{}/{}/*.png".format(self.gen, i))  # your image path
            for myFile in files2:
                image = cv2.imread(myFile)
                im_vae2.append(image)
            im_vae2 = np.array(im_vae2, dtype="float32")

            # convert integer to floating point values
            images2 = im_vae2.astype("float32")

            # resize images
            images2 = self.scale_images(images2, (299, 299, 3))
            print("Scaled", images1.shape, images2.shape)

            # pre-process images
            images2 = preprocess_input(images2)

            # calculate fid
            fid[i] = self.calculate_fid(self.model, images1, images2)
            print("FID: {}".format(fid[i]))

        print("Mean: {}, Std: {}".format(np.mean(fid), np.std(fid)))
        return fid


def gen_gt_imgs(dataset_name, xtrain):
    """Save Ground-truth images on HDD for use in FID scores"""
    if not os.path.exists("gt_imgs/{}".format(dataset_name)):
        os.makedirs("gt_imgs/{}".format(dataset_name))
        print("Saving Ground-Truth Images")
        for i, sample_batched in enumerate(xtrain):
            xt = sample_batched[0].numpy()
            for j in range(xt.shape[0]):
                imageio.imwrite(
                    "gt_imgs/{}/gt_img{}{}.png".format(dataset_name, i, j),
                    img_as_ubyte(convert_to_imshow_format(xt[j, :, :, :])),
                )
            if i == 16:  # stop after printing 8k images
                break
        print("GT images saved in: gt_imgs/{}\n".format(dataset_name))


def simplex_coordinates1(m):
    # *****************************************************************************
    # https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.py

    x = np.zeros([m, m + 1], dtype="float32")

    for k in range(0, m):
        #
        #  Set X(K,K) so that sum ( X(1:K,K)^2 ) = 1.
        #
        s = 0.0
        for i in range(0, k):
            s = s + x[i, k] ** 2

        x[k, k] = np.sqrt(1.0 - s)
        #
        #  Set X(K,J) for J = K+1 to M+1 by using the fact that XK dot XJ = - 1 / M.
        #
        for j in range(k + 1, m + 1):
            s = 0.0
            for i in range(0, k):
                s = s + x[i, k] * x[i, j]

            x[k, j] = (-1.0 / float(m) - s) / x[k, k]

    return x


def interpolate_along_clusters(point1, point2, n_steps):
    """Interpolates between two points in latent space and returns the interpolated points"""
    # Interpolate between the two latent points
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * point1.detach().numpy() + ratio * point2.detach().numpy()
        vectors.append(v)
    return np.asarray(vectors)


def assign_clusters(Z, u=None, phic=None):

    if Z is not None:
        Z = Z.detach().cpu().numpy()
        Z_sign = np.sign(Z)
        N, k = Z.shape[0], Z.shape[1] + 1
    elif u is not None and phic is not None:
        Z = (phic @ u).detach().cpu().numpy()
        Z_sign = np.sign(Z)
    else:
        return None
    N, k = Z.shape[0], Z.shape[1] + 1

    # get codebook
    codes, counts = np.unique(Z_sign, axis=0, return_counts=True)
    ind = np.flip(np.argsort(counts))
    ind = ind[:k]
    codes = codes[ind]
    if codes.shape[0] < k:
        return None
    else:
        # Assign hard clustering
        cl = np.zeros([N, k], dtype=int)
        for i in range(N):
            Hamdist = np.zeros(k)
            for j in range(k):
                Hamdist[j] = np.sum(np.abs(Z[i, :] - codes[j, :]))
            cl[i, np.argmin(Hamdist)] = 1

        return cl, Z


def ams(u, phic):
    cl, Z = assign_clusters(u, phic)
    N, k = Z.shape[0], Z.shape[1] + 1
    if cl is not None:

        # Calculate prototypes
        Sp = cl.T @ Z / np.sum(cl, axis=0, keepdims=True).T

        # Calculate cos_dist
        if k == 2:
            dist = np.concatenate(
                [np.sqrt((Z - Sp[0]) ** 2), np.sqrt((Z - Sp[1]) ** 2)], axis=1
            )
        else:
            dist = 1 - Z @ Sp.T / np.outer(
                np.sqrt(np.sum(Z**2, axis=1)), np.sqrt(np.sum(Sp**2, axis=1))
            )

        # Calculate cluster memberships
        dj = np.ones((N, k))
        for j in range(k):
            for c in range(k):
                if c != j:
                    dj[:, j] = dj[:, j] * dist[:, j]
        cm = dj / dj.sum(axis=1, keepdims=True)

        # Calculate Average Membership Score
        ams_score = (cm * cl / cl.sum(axis=0)).sum() / k
    else:
        ams_score = 0

    return ams_score
