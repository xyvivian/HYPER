import argparse
import ast
import os
import random
import time

import joblib
import numpy as np
import torch
import tqdm
from pyod.utils.utility import standardizer
from deepsets import DeepSet

from HNAEtrainer import HNET_trainer
from utils import PyODDataset, pyod_mlp_loader2, str2bool, sample_hps

# general settings
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--device', type=str, default='cuda', help="cuda or cpu")
parser.add_argument('--gpu_num', type=int, default=1, help="which gpu to use")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate")
parser.add_argument('--max_layer', type=int, default=8, help="maximum number of layers")
parser.add_argument('--min_layer', type=int, default=2, help="minimum number of layers")
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--threshold', type=int, default=1, help='minimum number of neurons per layer')
parser.add_argument('--use_bias', type=str2bool, default="True", help="Flag to use bias in AE")
parser.add_argument('--pe_size', type=int, default=4, help='Positional encoding dimension')
parser.add_argument('--pe_m', type=float, default=0.1, help='Positional encoding val')
parser.add_argument('--use_batch_norm', type=str2bool, default="False", help='Batch norm')
parser.add_argument('--avg_step', type=int, default=20)
parser.add_argument('--exp_num', type=int, default=0)

args = parser.parse_args()

# import f_val
f_reg = joblib.load(os.path.join('f-train', 'f-default.pkl'))
ds_model = torch.load(os.path.join('f-train', 'deepset_model.pt'))
ds_model.eval()

# define sampling parameters -> for demo just use small numbers
sample_size = 5
local_sample_size = 10
hn_iters = 20
sigma_size = 10
patience = 3

# define a list of files to run (we choose some fast ones for demo)
output_files = ['Cardiotocography', 'HeartDisease', 'Pima', 'WBC', 'WDBC']

global_best_roc_list = []
global_best_f_list = []
global_best_hp_list = []
global_best_std_list = []
global_time = []

for i, file in tqdm.tqdm(enumerate(output_files)):

    print('processing', i, file)

    mat_file = file

    # read the data
    X = np.genfromtxt(os.path.join('datasets', mat_file + '_X.csv'), delimiter=',')
    y = np.genfromtxt(os.path.join('datasets', mat_file + '_y.csv'), delimiter=',')

    n_samples = X.shape[0]
    n_features = X.shape[1]

    train_perc = 1

    total_ind = np.arange(len(y))
    random_state = np.random.RandomState(args.random_seed)

    random_state.shuffle(total_ind)
    train_ind = total_ind[0:int(n_samples * train_perc)]

    X_train, y_train = X[train_ind, :], y[train_ind]
    X_train, scalar = standardizer(X_train, keep_scalar=True)

    X_train = X_train.astype(np.float32)
    train_set = PyODDataset(X=X_train, y=y_train)
    train_loader = pyod_mlp_loader2(train_set, args)

    num_pixel = X_train.shape[1]
    best_ff = 0
    patience_counter = 0

    #### Initial range of HP+Architectures
    train_dropout = [0, 0.2]
    train_weight_decay = [0.0, 1e-5]
    train_compression_lst = np.arange(1.0, 2.0, 0.4)

    start = time.time()

    # initialize model
    model = HNET_trainer(num_pixel,
                         args,
                         hn_hiddn=[200, 200],
                         hn_dropout=0.2,
                         )

    # train with the initial HPs
    total_loss_trajectory, ind_loss_trajectory, auroc_lst, pred_recons_lst = \
        model.schedule_training(train_loader,
                                train_compression_lst=train_compression_lst,
                                eval_compression_lst=train_compression_lst,
                                epoch_save_step=args.avg_step,
                                train_dropout=train_dropout,
                                eval_dropout=train_dropout,
                                train_weight_decay=train_weight_decay,
                                eval_weight_decay=train_weight_decay)

    # local sample and retraining
    best_roc_list = []
    best_f_list = []
    best_hp_list = []
    best_sigma_list = []

    # initial range 10% of the HP range -> set to global best in the first place
    decay_mu, decay_std = random.uniform(1, 4), 0.2
    dropout_mu, dropout_std = random.uniform(0, 0.99), 0.01
    wd_mu, wd_std = random.uniform(0, 0.2), 0.01

    print(decay_mu, dropout_mu, wd_mu)

    for hn_iter in range(hn_iters):

        sample_decay, sample_dropout, sample_wd = sample_hps(decay_mu, decay_std, dropout_mu, dropout_std, wd_mu,
                                                             wd_std, sample_size)

        print('iter', hn_iter, len(sample_decay) * len(sample_dropout) * len(sample_wd))

        total_loss_trajectory, ind_loss_trajectory, auroc_lst, pred_recons_lst = \
            model.schedule_training(train_loader,
                                    train_compression_lst=sample_decay,
                                    eval_compression_lst=sample_decay,
                                    epoch_save_step=args.avg_step,
                                    train_dropout=sample_dropout,
                                    eval_dropout=sample_dropout,
                                    train_weight_decay=sample_wd,
                                    eval_weight_decay=sample_wd)

        # extract info from the HN
        trained_samples = []
        trained_samples_roc = []
        trained_samples_scores = []

        # use all early stopping results
        for u in range(len(auroc_lst)):
            curr_dict = auroc_lst[u]
            curr_dict_scores = pred_recons_lst[u]

            for key in curr_dict.keys():
                # print(i, j, key)
                key_split = key.split("|")
                arch_list = ast.literal_eval(key_split[0])
                depth = len(arch_list)
                compression = 1
                if depth > 1:
                    compression = arch_list[1] / arch_list[0]
                input_dim = arch_list[0]

                trained_samples.append([input_dim, compression, depth, float(key_split[1]), float(key_split[2])])
                trained_samples_roc.append(curr_dict[key])
                trained_samples_scores.append(curr_dict_scores[key].tolist())

        assert (len(trained_samples) == len(trained_samples_roc))
        assert (len(trained_samples) == len(trained_samples_scores))

        # score deepset features
        score_mat = np.asarray(trained_samples_scores).T
        n_samples = score_mat.shape[0]
        n_hp_configs = score_mat.shape[1]

        _, rep = ds_model(torch.Tensor(score_mat).view(n_samples, n_hp_configs, 1).to(args.device))
        score_rep = rep.cpu().detach().numpy()
        hp_rep = np.asarray(trained_samples)
        fh_score = np.load('f-train/data_reps_max.npy')

        model_features = np.repeat(fh_score[i, :].reshape(1, -1), n_hp_configs, axis=0)
        hp_features_scores = np.concatenate([hp_rep, model_features, score_rep], axis=1)

        pred_scores = f_reg.predict(hp_features_scores)
        best_idx = np.argmax(pred_scores)
        print('HP range', len(pred_scores))
        best_f = np.max(pred_scores)
        best_roc = trained_samples_roc[best_idx]
        best_hp = trained_samples[best_idx]

        best_roc_list.append(best_roc)
        best_f_list.append(best_f)
        best_hp_list.append(best_hp)
        print('iter', hn_iter, np.round(best_f, decimals=4), np.round(best_roc, decimals=4), best_hp)

        if best_f > best_ff:
            best_ff = best_f
        else:
            patience_counter += 1
            if patience_counter == patience - 1:
                print(i, file, hn_iter, hn_iter, hn_iter, hn_iter, hn_iter)
                break

        #### HPO loop Simple Version
        ########
        curr_hp = best_hp_list[-1]
        # compression rate needs to be reverted
        decay_mu, dropout_mu, wd_mu = 1 / curr_hp[1], curr_hp[3], curr_hp[4]

        # update sigma
        valid_loss = []
        temp_decay_std_list = []
        temp_dropout_std_list = []
        temp_wd_std_list = []

        # generate choices of sampling range
        for t in range(sigma_size):
            temp_decay_std = random.uniform(1, 3)
            temp_dropout_std = random.uniform(0, 0.5)
            temp_wd_std = random.uniform(0, 0.1)

            temp_decay_std_list.append(temp_decay_std)
            temp_dropout_std_list.append(temp_dropout_std)
            temp_wd_std_list.append(temp_wd_std)

        valid_loss_list = []
        # for each of the sampling range, get the average performance
        for t in range(sigma_size):
            eval_decay, eval_dropout, eval_wd = sample_hps(decay_mu, temp_decay_std_list[t], dropout_mu,
                                                           temp_dropout_std_list[t], wd_mu, temp_wd_std_list[t],
                                                           local_sample_size)
            aurocs, pred_recons = model.evaluate(train_loader,
                                                 eval_decay,
                                                 eval_dropout=eval_dropout,
                                                 eval_weight_decay=eval_wd)

            # extract info from the HN
            trained_samples = []
            trained_samples_scores = []

            curr_dict_scores = pred_recons

            for key in curr_dict_scores.keys():
                # print(i, j, key)
                key_split = key.split("|")
                arch_list = ast.literal_eval(key_split[0])
                depth = len(arch_list)
                compression = 1
                if depth > 1:
                    compression = arch_list[1] / arch_list[0]
                input_dim = arch_list[0]

                trained_samples.append([input_dim, compression, depth, float(key_split[1]), float(key_split[2])])
                trained_samples_scores.append(curr_dict_scores[key].tolist())

            assert (len(trained_samples) == len(trained_samples_scores))

            # score deepset features
            score_mat = np.asarray(trained_samples_scores).T
            n_samples = score_mat.shape[0]
            n_hp_configs = score_mat.shape[1]

            _, rep = ds_model(torch.Tensor(score_mat).view(n_samples, n_hp_configs, 1).to(args.device))
            score_rep = rep.cpu().detach().numpy()
            hp_rep = np.asarray(trained_samples)
            fh_score = np.load('f-train/data_reps_max.npy')

            model_features = np.repeat(fh_score[i, :].reshape(1, -1), n_hp_configs, axis=0)
            hp_features_scores = np.concatenate([hp_rep, model_features, score_rep], axis=1)

            pred_scores = f_reg.predict(hp_features_scores)

            valid_loss_list.append(
                np.mean(pred_scores) + temp_decay_std_list[t] * temp_dropout_std_list[t] * temp_wd_std_list[t])

        best_std = np.argmax(valid_loss_list)
        decay_std, dropout_std, wd_std = temp_decay_std_list[best_std], temp_dropout_std_list[best_std], \
            temp_wd_std_list[best_std]

        print(hn_iter, valid_loss_list, best_std)
        best_sigma_list.append([decay_std, dropout_std, wd_std])
        print('iter', hn_iter, 'updated HP to', curr_hp)
        print('iter', hn_iter, 'updated decay std, dropout std, wd_std to', decay_std, dropout_std, wd_std)

    # the best performances are recorded here
    global_time.append(time.time() - start)
    global_best_roc_list.append(best_roc_list)
    global_best_f_list.append(best_f_list)
    global_best_hp_list.append(best_hp_list)
    global_best_std_list.append(best_sigma_list)

# show the final results
for i, file in enumerate(output_files):
    print(file, 'selected model ROC', global_best_roc_list[i][np.argmax(global_best_f_list[i])])
