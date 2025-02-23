import torch
from torch.utils.data import (
    DataLoader,
    Subset,
    ConcatDataset,
    random_split,
    SubsetRandomSampler,
)
from utils import (
    create_dirs,
    convert_to_imshow_format,
    select_cluster,
    assign_soft_clusters,
    ams,
)
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from soft_ksc_rkm_model import *
import logging
import argparse
import time
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import numpy as np
import optuna
from optuna import Trial, visualization
from sklearn.metrics import normalized_mutual_info_score as nmi
import json
import optuna.visualization as vis
import wandb

# ==================================================================================================================


def train_model(rkm, xtrain_loader, xval_loader, opt):

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    ct = time.strftime("%Y%m%d-%H%M")
    filename = "k_" + str(opt.k) + "_cluster_" + str(opt.cluster) + "_" + ct
    dirs = create_dirs(name=opt.dataset_name, ct=filename)
    dirs.create()

    # noinspection PyArgumentList
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(
                "log/{}/{}_{}.log".format(opt.dataset_name, opt.dataset_name, ct)
            ),
            logging.StreamHandler(),
        ],
    )

    device = torch.device(opt.proc)
    ngpus = torch.cuda.device_count()

    rkm = rkm.to(device)

    # xtrain_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.mb_size, shuffle=opt.shuffle, num_workers=opt.workers)
    # xval_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.mb_size, shuffle=opt.shuffle, num_workers=opt.workers)
    logging.info(rkm)
    logging.info(opt)
    logging.info("\nN: {}, mb_size: {}".format(len(xtrain_loader.dataset), opt.mb_size))
    logging.info("We are using {} GPU(s)!".format(ngpus))

    # Accumulate trainable parameters in 2 groups:
    # 1. Manifold_params 2. Network params
    param_g, param_e1 = param_state(rkm)

    optimizer1 = stiefel_opti(param_g, opt.lrg)
    optimizer2 = torch.optim.AdamW(param_e1, lr=opt.lr, weight_decay=0)

    # Train =========================================================================================================
    start = datetime.now()
    Loss_stk = np.empty(shape=[0, 4])
    Val_Loss_stk = np.empty(shape=[0, 4])
    Loss_clstr = np.empty(shape=[0, 1])
    cost, l_cost = np.inf, np.inf  # Initialize cost

    # initialize validation loss
    val_cluster_cost, l_val_cluster_cost = np.inf, np.inf
    # number of epochs to wait to increase number of clusters if no improvement
    patience = 50
    current_patience = 0
    decoder_training = True
    is_best = False
    t = 1

    while (
        cost > -1e10 and t <= opt.max_epochs
    ):  # run epochs until convergence or cut-off
        avg_loss, avg_f1, avg_f2, avg_f3 = 0, 0, 0, 0
        nb = 0
        rkm.train()  # Set the model to training mode

        for _, sample_batched in enumerate(
            tqdm(xtrain_loader, desc="Epoch {}/{}".format(t, opt.max_epochs))
        ):

            f1, f2, f3, f4, _, _, _ = rkm(sample_batched[0].to(device))

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            if t < opt.recon_epochs:
                loss = f1 + f2 + f4
            # elif decoder_training:
            #    loss = f2
            else:
                loss = f1 + f2 + f3 + f4
                # loss = f3

            loss.backward()
            optimizer2.step()
            optimizer1.step()

            avg_loss += loss.item()
            avg_f1 += f1.item()
            avg_f2 += f2.item()
            avg_f3 += f3.item() + f4.item()

            nb += 1

        cost = avg_loss / nb
        avg_f1 = avg_f1 / nb
        avg_f2 = avg_f2 / nb
        avg_f3 = avg_f3 / nb

        # Validation phase
        val_avg_loss, val_avg_f1, val_avg_f2, val_avg_f3 = 0, 0, 0, 0
        val_nb = 0
        rkm.eval()
        with torch.no_grad():
            for _, val_batch in enumerate(tqdm(xval_loader, desc="Validation")):
                val_f1, val_f2, val_f3, val_f4, _, _, _ = rkm(val_batch[0].to(device))

                if t < opt.recon_epochs:
                    val_loss = val_f1 + val_f2 + val_f4
                # elif decoder_training:
                #    val_loss = val_f2
                else:
                    val_loss = val_f1 + val_f2 + val_f3 + val_f4

                val_avg_loss += val_loss.item()
                val_avg_f1 += val_f1.item()
                val_avg_f2 += val_f2.item()
                val_avg_f3 += val_f3.item() + val_f4.item()

                val_nb += 1

        # Calculate the validation loss and other metrics
        val_cost = val_avg_loss / val_nb
        val_avg_f1 = val_avg_f1 / val_nb
        val_avg_f2 = val_avg_f2 / val_nb
        val_avg_f3 = val_avg_f3 / val_nb

        logging.info(
            "Epoch {}/{}, Loss: [{}], Kpca: [{}], Recon: [{}], Cluster: [{}]".format(
                t, opt.max_epochs, cost, avg_f1, avg_f2, avg_f3
            )
        )
        Loss_stk = np.append(Loss_stk, [[cost, avg_f1, avg_f2, avg_f3]], axis=0)
        Val_Loss_stk = np.append(
            Val_Loss_stk, [[val_cost, val_avg_f1, val_avg_f2, val_avg_f3]], axis=0
        )
        # Loss_clstr = np.append(Loss_clstr, [Loss_clusters.tolist()], axis=0)

        # Check if the validation loss has improved
        if val_cost < l_cost:
            l_cost = min(val_cost, l_cost)
            is_best = True
            # save best model
            dirs.save_checkpoint(
                {
                    "epochs": t,
                    "rkm_state_dict": rkm.state_dict(),
                    "optimizer1": optimizer1.state_dict(),
                    "optimizer2": optimizer2.state_dict(),
                    "Loss_stk": Loss_stk,
                    "Val_Loss_stk": Val_Loss_stk,
                    "Loss_clstr": Loss_clstr,
                },
                is_best,
            )
            current_patience = 0  # reset patience
        else:
            current_patience += 1

        # If no improvement found for patience epochs, switch to decoder training phase
        if current_patience >= patience:
            if decoder_training:
                break
            print(
                "No improvement for {}. Switching to decoder training".format(patience)
            )
            # Perform SVD on full dataset:
            all_data = torch.cat([batch[0] for batch in xtrain_loader], dim=0)

            # rkm.svd(all_data.to(device))

            # Fix the encoder and Ut weights
            for param in rkm.encoder.parameters():
                param.requires_grad = False
            for param in param_g:
                param.requires_grad = False

            decoder_training = True
            current_patience = (
                0  # reset patience and increase epochs for decoder training
            )
            patience = patience * 2

        t += 1

    logging.info(
        "Finished Training. Lowest cost: {}"
        "\nLoading best checkpoint [{}] & computing sub-space...".format(
            l_cost, dirs.dircp
        )
    )

    # Load Checkpoint
    sd_mdl = torch.load("cp/{}/{}".format(opt.dataset_name, dirs.dircp))
    rkm.load_state_dict(sd_mdl["rkm_state_dict"])

    # Compute final features ========================================================================================

    all_data = torch.cat([batch[0] for batch in xtrain_loader], dim=0)
    # val_data = torch.cat([batch[0] for batch in xval_loader], dim=0)
    # val_targets = torch.cat([batch[1] for batch in xval_loader], dim=0)

    val_data_list = []
    val_targets_list = []

    for batch in xval_loader:
        val_data_list.append(batch[0])
        val_targets_list.append(batch[1])

    val_data = torch.cat(val_data_list, dim=0)
    val_targets = torch.cat(val_targets_list, dim=0)

    all_data = torch.cat([all_data, val_data], dim=0)
    # all_data = ConcatDataset([train_data, val_data])
    #
    # xall_loader = torch.utils.data.DataLoader(all_data, batch_size=len(all_data), shuffle=opt.shuffle,
    #                                             num_workers=opt.workers)
    # xval_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=opt.shuffle,
    #                                          num_workers=opt.workers)
    with torch.no_grad():
        # rkm.compute_eval_centering_term(all_data.to(device))
        # Final forward pass with all data
        rkm.train()
        _, _, _, _, phi, e, h = rkm(all_data.to(device))

        # Compute final validation loss
        rkm.eval()
        # val_f1, val_f2, val_f3, val_f4, _, _, _ = rkm(next(iter(xval_loader))[0].to(device))
        # val_f1, val_f2, val_f3, val_f4, _, _, _ = rkm(val_data.to(device),trained_phi=phi)
        val_f1, val_f2, val_f3, val_f4, _, eval, _ = rkm(val_data.to(device))
        val_cost = val_f1 + val_f2 + val_f3 + val_f4
        # wandb.log({'KSC Loss': val_f1.item(), 'Recon Loss': val_f2.item(), 'Cluster Loss': val_f3.item(), 'Total Loss': val_cost.item()})

    # Assign soft clusters ==========================================================================================
    cluster_predictions, _ = assign_soft_clusters(e, opt.k)
    cluster_predictions_val, _ = assign_soft_clusters(eval, opt.k)
    try:
        ams_score = ams(eval, opt.k)
    except Exception as ex:
        ams_score = 0
    if val_targets.dim() > 1:
        val_targets = val_targets.argmax(dim=1)
    nmi_score = nmi(cluster_predictions_val.numpy(), val_targets.numpy())

    # Save Model and Tensors ======================================================================================
    torch.save(
        {
            "rkm": rkm,
            "rkm_state_dict": rkm.state_dict(),
            "optimizer1": optimizer1.state_dict(),
            "optimizer2": optimizer2.state_dict(),
            "Loss_stk": Loss_stk,
            "Val_Loss_stk": Val_Loss_stk,
            "Loss_clstr": Loss_clstr,
            "opt": opt,
            "h": h,
            "e": e,
            "Ut": rkm.Ut,
            "phi": phi,
            "cluster_predictions": cluster_predictions,
        },
        "out/{}/{}".format(opt.dataset_name, dirs.dirout),
    )
    logging.info("\nSaved File: {}".format(dirs.dirout))

    return rkm, val_cost, cluster_predictions, ams_score, nmi_score, ct


def objective(trial: Trial, train_data, val_data, opt):

    # Set hyperparameters
    # lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    lr = 0.0001
    lrg = 0.0001
    # lrg = trial.suggest_float('lrg', 1e-5, 1e-3, log=True)
    k = trial.suggest_int("k", 2, 10, 1)
    # capacity = trial.suggest_int('capacity', 32, 256, 10)
    # mb_size = trial.suggest_int('mb_size', 128, 512, 128)
    mb_size = 256

    c_ksc = trial.suggest_float("c_ksc", 0.001, 1, log=True)
    # c_accu = trial.suggest_float('c_accu', 0.01, 1, log=True)
    c_accu = 1 - c_ksc
    c_clust = trial.suggest_float("c_clust", 0.001, 1, log=True)

    # c_accu = trial.suggest_float('c_accu', 0, 1-c_ksc)
    # c_clust = 1-c_accu - c_ksc
    # trial.set_user_attr('c_clust', c_clust)
    # c_clust = trial.suggest_float('c_clust', 0, 1)

    c_balance = trial.suggest_float("c_balance", 0, 1)
    # recon_epochs = trial.suggest_int('recon_epochs', 10, 80)
    recon_epochs = 32

    # Set hyperparameters
    opt.lr = lr
    opt.lrg = lrg
    opt.mb_size = mb_size
    opt.k = k
    # opt.capacity = capacity
    opt.c_ksc = c_ksc
    opt.c_accu = c_accu
    opt.c_clust = c_clust
    opt.c_balance = c_balance
    opt.recon_epochs = recon_epochs

    ipVec_dim = 28 * 28
    nChannels = 1

    rkm = RKM_Stiefel(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels)
    rkm.to(device=opt.proc)

    # Train model
    rkm, val_cost, cluster_predictions, ams_score, nmi_score, ct = train_model(
        rkm, train_data, val_data, opt
    )

    # Monitor average membership score and NMI

    trial.set_user_attr("ams_score", ams_score.item())
    trial.set_user_attr("nmi_score", nmi_score)
    trial.set_user_attr("ct", ct)
    # Log the hyperparameters and the result
    wandb.log(
        {
            "k": k,
            "c_ksc": c_ksc,
            "c_accu": c_accu,
            "c_clust": c_clust,
            "c_balance": c_balance,
            "recon_epochs": recon_epochs,
            "lr": lr,
            "lrg": lrg,
            "mb_size": mb_size,
            "val_cost": val_cost,
            "ams_score": ams_score,
            "nmi_score": nmi_score,
        }
    )
    return ams_score


def tune_model(train_data, val_data, opt, n_trials=100):

    study_name = "genksc_study_" + opt.dataset_name

    wandb.init(project=study_name)
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///rkm.db",
        study_name=study_name,
        load_if_exists=True,
    )

    objective_rkm = lambda trial: objective(trial, train_data, val_data, opt)
    study.optimize(objective_rkm, n_trials=n_trials)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("Trial Number: ", trial.number)
    print("Time: ", trial.user_attrs.get("ct", "N/a"))
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save the best parameters to a JSON file
    json_filename = "best_params.json_{timestamp}".format(
        timestamp=datetime.now().strftime("%Y%m%d-%H%M")
    )
    with open(json_filename, "w") as json_file:
        json.dump(trial.params, json_file)

    wandb.finish()


def main():

    # Model Settings =================================================================================================
    parser = argparse.ArgumentParser(
        description="St-RKM Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mnist",
        help="Dataset name: mnist/mnist012/mnist0123/fashion-mnist/fashion-mnist0123/svhn/dsprites/3dshapes/cars3d/gaussians",
    )
    parser.add_argument(
        "--h_dim", type=int, default=40, help="Dim of latent vector (>=k-1)"
    )
    parser.add_argument("--k", type=list, default=10, help="Number of initial clusters")
    parser.add_argument(
        "--capacity", type=int, default=64, help="Conv_filters of network"
    )
    parser.add_argument(
        "--mb_size", type=int, default=256, help="Mini-batch size"
    )  # 256
    parser.add_argument(
        "--x_fdim1", type=int, default=1024, help="Input x_fdim1"
    )  # 256 for MNIST
    parser.add_argument("--x_fdim2", type=int, default=50, help="Input x_fdim2")
    parser.add_argument(
        "--inter_fact",
        type=float,
        default=1,
        help="factor [0,1] to interpolate between KPCA(0) and KSC(1)",
    )
    parser.add_argument(
        "--c_ksc", type=float, default=0.99, help="eta hyperparameter"
    )  # used for KPCA term
    parser.add_argument(
        "--c_accu", type=float, default=0.001, help="Input weight on recons_error"
    )
    parser.add_argument(
        "--c_clust", type=float, default=0.009, help="Input weight on clust_error"
    )
    parser.add_argument(
        "--c_balance", type=float, default=0.05, help="Weight on cluster balance term"
    )  # 0 is no balance, 1 is full balance
    parser.add_argument(
        "--c_stab", type=float, default=1, help="Stabilization parameter for kpca term"
    )
    parser.add_argument("--noise_level", type=float, default=1e-3, help="Noise-level")
    parser.add_argument(
        "--loss",
        type=str,
        default="deterministic",
        help="loss type: deterministic/noisyU/splitloss",
    )

    # Training Settings =============================
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Input learning rate for ADAM optimizer"
    )  # for MNIST 2e-4
    parser.add_argument(
        "--lrg",
        type=float,
        default=1e-4,
        help="Input learning rate for Cayley_ADAM optimizer",
    )  # for MNIST 1e-4
    parser.add_argument(
        "--max_epochs", type=int, default=150, help="Input max_epoch"
    )  # normally 250
    parser.add_argument(
        "--recon_epochs",
        type=int,
        default=32,
        help="Input recon_epoch: amount of epochs before cluster loss is used",
    )
    parser.add_argument(
        "--proc", type=str, default="cuda", help="device type: cuda or cpu"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=False, help="shuffle dataset: True/False"
    )
    parser.add_argument(
        "--seed", type=int, default=3, help="Torch and Numpy random seed"
    )
    parser.add_argument("--level", type=int, default=0, help="Hierarchical level")
    parser.add_argument("--cluster", type=str, default="0", help="Cluster number")
    # parser.add_argument('--select_MNIST_subset', type=np.array, default=None, help='select subset of MNIST dataset')

    opt = parser.parse_args()

    # Load Data ======================================================================================================
    # Load data

    # import os
    # os.environ['TORCH_BOTTLENECK'] = '1'
    """ Load Training Data """
    xtrain_loader, ipVec_dim, nChannels, xtest = get_dataloader(args=opt)

    # dataset = FashionMNIST(root='data/', train=True, download=True, transform=ToTensor())
    # Split initial data into train and validation
    # ipVec_dim = 28 * 28
    # nChannels = 1

    n_train = int(len(xtrain_loader.dataset) * 0.8)
    n_val = len(xtrain_loader.dataset) - n_train

    # Define the size of the validation set
    validation_split = 0.2
    dataset_size = len(xtrain_loader.dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)

    # Split the indices into training and validation sets
    train_indices, val_indices = indices[split:], indices[:split]

    # Create sampler objects using SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Create DataLoader objects for training and validation
    xtrain_loader = DataLoader(
        xtrain_loader.dataset,
        batch_size=xtrain_loader.batch_size,
        sampler=train_sampler,
    )
    xval_loader = DataLoader(
        xtrain_loader.dataset, batch_size=xtrain_loader.batch_size, sampler=val_sampler
    )
    # train_data, val_data = torch.utils.data.random_split(xtrain_loader.dataset, [n_train, n_val])

    """ Load Model """

    #
    # # Load best params from JSON file
    # with open('best_params.json_20240305-0336', 'r') as json_file:
    #     best_params = json.load(json_file)
    #     opt.lr = best_params['lr']
    #     #opt.c_accu = best_params['c_accu']
    #     #opt.c_clust = best_params['c_clust']
    #     #opt.c_balance = best_params['c_balance']
    #     #opt.recon_epochs = best_params['recon_epochs']
    #     opt.mb_size = best_params['mb_size']
    #     opt.lrg = best_params['lrg']
    #     opt.mb_size = 256
    #

    #
    # rkm = RKM_Stiefel(ipVec_dim=ipVec_dim, args=opt, nChannels=nChannels)
    # rkm.to(device = opt.proc)
    # Run forward pass with profiler

    # with torch.autograd.profiler.profile(use_cuda=True) as prof:

    # output = rkm(train_data.dataset.data[:2].to(device=opt.proc))

    # trained_rkm, val_cost, cluster_predictions, ams_score, nmi_score, _ = train_model(rkm, xtrain_loader, xval_loader, opt)

    # print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    # torch.autograd.profiler.visualize(prof)
    # Set up Weight and Biases

    tune_model(train_data=xtrain_loader, val_data=xval_loader, opt=opt, n_trials=32)

    # try:
    #     tune_model(rkm, train_data, val_data, opt)
    # except Exception as e:
    #     opt.max_epochs = 300
    #     opt.recon_epochs = 100
    #     trained_rkm, val_cost, cluster_predictions, ams_score = train_model(rkm, train_data, val_data, opt)
    #     print(e)
    study = optuna.load_study(
        study_name="genksc_study_" + opt.dataset_name, storage="sqlite:///rkm.db"
    )

    # print(study.trials[0].user_attrs.get('ct', 0.0))
    # import matplotlib
    # matplotlib.use('Agg')
    # optuna.visualization.plot_optimization_history(study)
    # print("Loss: {:.3f} AMS_score: {:.8f} Time {}".format(study.trials[15].value, study.trials[15].user_attrs.get('ams_score', 0.0), study.trials[15].user_attrs.get('ct', 0.0)))
    # print(study.trials[15].params)
    ct_files2 = []
    k_files2 = []
    for trial in study.trials:
        # print("Loss: {:.3f} AMS_score: {:.8f} NMI_score {} Time {}".format(trial.value, trial.user_attrs.get('ams_score', 0.0), trial.user_attrs.get('nmi_score', 0.0), trial.user_attrs.get('ct', 0.0)))
        # print(trial.params)

        try:
            ct_files2.append(trial.user_attrs.get("ct", 0.0))
            k_files2.append(trial.params["k"])
        except:
            continue
    #
    # print(ct_files)
    # print(k_files)
    #
    # save ct_files and k_files to a json file
    with open("ct_files.json", "w") as json_file:
        json.dump(ct_files2, json_file)
    with open("k_files.json", "w") as json_file:
        json.dump(k_files2, json_file)


if __name__ == "__main__":
    main()
