import os

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
import numpy as np
import random
import torch
import torch.utils
import torch.utils.data
from torch import nn
import scipy.io as scio
import time
from model.model import PGMSU
from hyperVca import hyperVca
from loadhsi import loadhsi
import utilities
import hydra
from omegaconf import DictConfig
import logging
import math


log = logging.getLogger(__name__)

@hydra.main(version_base=None,config_path="config",config_name="config")
def main(cfg:DictConfig):
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            if not torch.cuda.is_available():
                print('WARNING: You have a CUDA device, so you should probably run with --cuda')
            else:
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True


    set_seed(seed)

    tic = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # cases = ['ex2', 'ridge']
    # case = cases[1]
    # load data
    # Y, A_true, P = loadhsi(case)
    DATASET = cfg.dataset
    DATASET_PATH = "../hsi_datasets/"
    DATSET_FILE = os.path.join(DATASET_PATH,DATASET)
    data = scio.loadmat(DATSET_FILE)
    Y = data['Y'].astype(np.float32)
    AAA = data['A'].astype(np.float32)
    MMM = data['M'].astype(np.float32)
    P = AAA.shape[0]
    H,W = map(int,list(data['HW'].ravel()))
    A_true = AAA.reshape(P,H,W)
    print(f"{Y.shape=} {A_true.shape=} {P=}")

    lambda_kl = cfg.kl
    lambda_sad = cfg.sad
    lambda_vol = cfg.vol

    # if case == 'ridge':
    #     lambda_kl = 0.1
    #     lambda_sad = 3
    #     lambda_vol = 7
    #
    # if case == 'ex2':
    #     lambda_kl = 0.1
    #     lambda_sad = 0
    #     lambda_vol = 0.5
    # if case == 'urban':
    #     lambda_kl = 0.001
    #     lambda_sad = 4
    #     lambda_vol = 6
    #

    Channel = Y.shape[0]
    N = Y.shape[1]
    batchsz = N//10
    # lr = 1e-3
    # epochs = 200
    # epochs = 2
    epochs = cfg.epochs
    lr = cfg.lr
    z_dim = 4

    model_weights = './PGMSU_weight/'
    output_path = './PGMSU_out/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(model_weights):
        os.makedirs(model_weights)
    model_weights += 'PGMSU.pt'


    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)


    Y = np.transpose(Y)
    train_db = torch.tensor(Y)
    train_db = torch.utils.data.TensorDataset(train_db)
    train_db = torch.utils.data.DataLoader(train_db, batch_size=batchsz, shuffle=True)

    EM, _, _ = hyperVca(Y.T, P)
    EM = EM.T
    EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype('float32')
    EM = torch.tensor(EM).to(device)

    model = PGMSU(P, Channel, z_dim).to(device)
    model.apply(weights_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    print('Start training!')
    for epoch in range(epochs):
        model.train()
        for step, y in enumerate(train_db):
            y = y[0].to(device)
            y_hat, mu, log_var, a, em_tensor = model(y)

            loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

            kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
            kl_div = kl_div.sum() / y.shape[0]
            # KL balance of VAE
            kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))

            if epoch < epochs // 2:
                # pre-train process
                loss_vca = (em_tensor - EM).square().sum() / y.shape[0]
                loss = loss_rec + lambda_kl * kl_div + 0.1 * loss_vca
            else:
                # training process
                # constrain 1 min_vol of EMs
                em_bar = em_tensor.mean(dim=1, keepdim=True)
                loss_minvol = ((em_tensor - em_bar) ** 2).sum() / y.shape[0] / P / Channel

                # constrain 2 SAD for same materials
                em_bar = em_tensor.mean(dim=0, keepdim=True)  # [1,5,198] [1,z_dim,Channel]
                aa = (em_tensor * em_bar).sum(dim=2)
                em_bar_norm = em_bar.square().sum(dim=2).sqrt()
                em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()

                sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
                loss_sad = sad.sum() / y.shape[0] / P
                loss = loss_rec + lambda_kl * kl_div + lambda_vol * loss_minvol + lambda_sad * loss_sad

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        # if (epoch + 1) % 50 == 0:
        #     torch.save(model.state_dict(), model_weights)
        #     scio.savemat(output_path + 'loss.mat', {'loss': losses})
        #     print('epoch:',epoch+1,' save results!')

    toc = time.time()
    print('time elapsed:', toc - tic)

    model.eval()
    with torch.no_grad():
        y_hat, mu, log_var, A, em_hat = model(torch.tensor(Y).to(device))
        A_hat = A.cpu().numpy().T
        A_true = A_true.reshape(P, N)
        dev = np.zeros([P, P])
        for i in range(P):
            for j in range(P):
                dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
        pos = np.argmin(dev, axis=0)

        A_hat = A_hat[pos, :]
        em_hat = em_hat[:, pos, :]

        armse_a = np.mean(np.sqrt(np.mean((A_hat - A_true) ** 2, axis=0)))

        Y_hat = y_hat.cpu().numpy()
        armse_y = np.mean(np.sqrt(np.mean((Y_hat - Y) ** 2, axis=1)))
        norm_y = np.sqrt(np.sum(Y ** 2, 1))
        norm_y_hat = np.sqrt(np.sum(Y_hat ** 2, 1))
        asad_y = np.mean(np.arccos(np.sum(Y_hat * Y, 1) / norm_y / norm_y_hat))

        # scio.savemat(output_path + 'results.mat', {'EM': em_hat.data.cpu().numpy(),
        #                                         'A': A_hat.T,
        #                                         'Y_hat': y_hat.cpu().numpy()})

        M_pred = em_hat.mean(axis=0).T.cpu().detach().numpy()
        print(f"{M_pred.shape=},{A_hat.shape=}")
        A_corrected,M_corrected = utilities.correct_permuation(A_pred=A_hat,A_true=AAA,M_pred=M_pred,M_true=MMM)
        OUTPUT_PATH = "outputs"
        save_path = os.path.join(OUTPUT_PATH,DATASET)
        os.makedirs(save_path,exist_ok=True)
        utilities.plot_figures(A_pred=A_corrected,A_true=AAA,M_pred=M_corrected,M_true=MMM,H=H,save_path=save_path)
        error_values = utilities.calculate_errors(A_pred=A_corrected,A_true=AAA,M_pred=M_corrected,M_true=MMM,save_path=save_path)
        log.info(error_values)
        log.info(cfg)
        val =  error_values["total_rmse"].item()
        if (math.isnan(val)):
            return 100

        return val

if __name__ == "__main__":
    main()
