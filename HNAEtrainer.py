import random

from sklearn.metrics import roc_auc_score

from HMLP import HMLP
from MLP import MLP, MLP_weight_shapes, MLP_Hcontainer
from utils import *


class HNET_trainer():
    def __init__(self,
                 num_pixel,
                 args,
                 hn_hiddn=[200, 200],
                 hn_dropout=0.2):
        super(HNET_trainer, self).__init__()
        self.num_pixel = num_pixel
        self.max_layer = args.max_layer
        self.min_layer = args.min_layer
        self.threshold = args.threshold
        self.device = args.device
        self.hn_hiddn = hn_hiddn
        self.hn_dropout = hn_dropout
        self.device = args.device
        self.pe_size = args.pe_size
        self.pe_m = args.pe_m
        self.learning_rate = args.learning_rate
        self.train_epochs = args.train_epochs
        self.use_batch_norm = args.use_batch_norm
        # initialize models
        self.init_models()

    def init_models(self,
                    verbose=True):
        """
        Model intialization, to enable faster batch training,
        We feed weights concurrently to the downstream DOD models (AEs),
        """
        # feed to several children MLPs -> enable the batch training regime
        self.ae_model_lst = []
        for cur_layer in range(self.min_layer, self.max_layer + 1, 2):
            ae_model = MLP(out_fn=torch.nn.Sigmoid(),
                           verbose=False,
                           use_batch_norm=self.use_batch_norm)
            self.ae_model_lst.append(ae_model)

        # feed the hidden dimensions to the h_container
        A, inpt, hiddn = generate_input_dim_lst(num_layer=self.max_layer,
                                                max_layer=self.max_layer,
                                                compression=1,
                                                input_dim=self.num_pixel,
                                                threshold=self.threshold)
        self.default_hiddn = hiddn[0:-1]
        h_container = MLP_Hcontainer(self.default_hiddn,
                                     pe_size=self.pe_size,
                                     pe_m=self.pe_m,
                                     dropout=0.0,
                                     weight_decay=0.0,
                                     device=self.device)
        # find the parameters of the maximum architecture to feed into the network
        max_param_shapes = MLP_weight_shapes(n_in=self.num_pixel,
                                             n_out=self.num_pixel,
                                             hidden_layers=self.default_hiddn)

        # build one hypernetwork - taken in hidden dimension as the cond in size
        self.hnet = HMLP(max_param_shapes,
                         cond_in_size=h_container.to_torch_tensor().shape[0],  # dimenions to encode the task
                         layers=self.hn_hiddn,
                         dropout_rate=self.hn_dropout).to(self.device)
        self.hnet.apply_hyperfan_init()

    def architecture_masking(self,
                             W_weights,
                             amask,
                             hidden_lst):
        ret_W_weights = []
        for i in range(len(amask)):
            cur_A = amask[i][0]
            prev_A = amask[i][1]
            cur_hidden = hidden_lst[i]
            # the weights should be ignored by the DOD models
            if cur_hidden == 0:
                continue
            else:
                A_matrix = torch.ones(W_weights[2 * i].shape, device=W_weights[2 * i].device, requires_grad=False)
                A_bias_matrix = torch.ones(W_weights[2 * i + 1].shape, device=W_weights[2 * i + 1].device,
                                           requires_grad=False)
                A_matrix[cur_A:, :] = 0.0
                A_matrix[:, prev_A:] = 0.0
                A_bias_matrix[cur_A:] = 0.0

                W_weights[2 * i] = W_weights[2 * i] * A_matrix
                W_weights[2 * i + 1] = W_weights[2 * i + 1] * A_bias_matrix
                ret_W_weights.append(W_weights[2 * i])
                ret_W_weights.append(W_weights[2 * i + 1])
        return ret_W_weights

    def build_architectures(self,
                            compression_lst,
                            verbose=True):
        """
        Generate architecture masking, hidden dimension lists and h_containers
        across different architectures.
        Args: list of desired weight compression factors 
        Return: (1) A masks: across different layer depths,layer widths 
                (2) Hidden dimension list: 
                (3) H container list: list of float tensors, representing the 
                    lambdas to feed into the Hypernetwork
        """
        # creates list for different number of layers
        A_list = []
        hidden_list = []
        h_list = []
        # from min layer to max layer
        for cur_layer in range(self.min_layer, self.max_layer + 1, 2):
            amasks = []
            hidden_dimensions = []
            h_container_list = []
            # different input decay
            for i in compression_lst:
                A, inpt, hiddn = generate_input_dim_lst(num_layer=cur_layer,
                                                        max_layer=self.max_layer,
                                                        compression=i,
                                                        input_dim=self.num_pixel,
                                                        threshold=self.threshold)
                h_container = MLP_Hcontainer(hiddn[0:-1],
                                             pe_size=self.pe_size,
                                             pe_m=self.pe_m,
                                             dropout=0.0,
                                             weight_decay=0.0,
                                             device=self.device)
                h_container_list.append(h_container)
                amasks.append(A)
                hidden_dimensions.append(hiddn)
            A_list.append(amasks)
            hidden_list.append(hidden_dimensions)
            h_list.append(h_container_list)
        return A_list, hidden_list, h_list

    def compute_individual_loss(self,
                                X,
                                A_list,
                                train_dropout,
                                train_weight_decay,
                                h_list,
                                hidden_list,
                                criterion,
                                used_l_num):
        batch_loss_lst = []
        # COMPUTE individual loss phase 
        for l_num, arch_matrix in enumerate(A_list):
            # layerwise individual losses
            layer_individual_loss_lst = []
            # iterate through all possible dropout values and weight decay values
            for dropout_ in train_dropout:
                for weight_decay_ in train_weight_decay:
                    # set the h matrix
                    h_layer_lst = h_list[l_num]
                    h_tensor_matrix = []
                    for h_ in h_layer_lst:
                        h_.set_H_container(dropout=dropout_,
                                           weight_decay=weight_decay_)
                        h_tensor_matrix.append(h_.to_torch_tensor())
                    h_matrix = torch.stack(h_tensor_matrix, dim=0).to(torch.float32).to(self.device)

                    W_batch = self.hnet.forward(cond_input=h_matrix)

                    for W_idx, W_weights in enumerate(W_batch):
                        ret_W_weights = self.architecture_masking(W_weights,
                                                                  amask=arch_matrix[W_idx],
                                                                  hidden_lst=hidden_list[used_l_num][W_idx])
                        pred = self.ae_model_lst[l_num].forward(X,
                                                                weights=ret_W_weights,
                                                                h_container=h_layer_lst[W_idx])[0]
                        individual_loss = criterion(pred, X)
                        layer_individual_loss_lst.append(individual_loss.item())

            batch_loss_lst.append(layer_individual_loss_lst)
        return batch_loss_lst

    def schedule_training(self,
                          train_loader,
                          train_compression_lst,
                          eval_compression_lst=None,
                          epoch_save_step=50,  # early saving the results
                          eval_loader=None,  # test data
                          train_dropout=[0.0, 0.2],  # dropout list
                          eval_dropout=[0.0, 0.2],  # evaluation dropout list
                          train_weight_decay=[0.0, 1e-5],  # train weight decay list
                          eval_weight_decay=[0.0, 1e-5],  # eval weight decay list
                          verbose=True  # Bool: if verbose, print and plot information
                          ):

        A_list, hidden_list, h_list = self.build_architectures(train_compression_lst)
        if eval_compression_lst is None:
            eval_compression_lst = train_compression_lst
        if eval_loader is None:
            eval_loader = train_loader

        # training procedure
        # Adam usually works well in combination with hypernetwork training.
        optimizer = torch.optim.Adam(self.hnet._internal_params, lr=self.learning_rate)
        criterion = nn.MSELoss()
        # total loss
        total_loss_trajectory = []
        # individual losses across different number of layers and different layer widths
        ind_loss_trajectory = []
        # early stop aurocs (if early stopping is used)
        auroc_lst = []
        pred_recons_lst = []

        for epoch in range(self.train_epochs):

            # For each epoch.
            # Iterate over the whole MNIST/FashionMNIST training set.
            # Note, that both datasets have the same number of training samples.
            loss_lst = []
            ind_loss_lst = []

            for data in train_loader:

                # Current mini-batch of MNIST samples.
                X = data[0].to(self.device)
                optimizer.zero_grad()
                # for each individual_losses
                loss = torch.zeros(1).to(self.device)
                loss.requires_grad = True

                # architectures to update: from largest number of layers to smallest number of layers
                used_num = len(A_list) - 1 - int(epoch / (self.train_epochs / len(A_list)))

                # sample a dropout rate
                sampled_dropout = random.choice(train_dropout)
                sampled_weight_decay = random.choice(train_weight_decay)

                # calculate total losses from individual models
                for used_l_num in range(used_num, len(A_list)):
                    arch_matrix = A_list[used_l_num]

                    # Create batched h_containers to feed into each architecture
                    h_layer_lst = h_list[used_l_num]
                    h_tensor_matrix = []
                    for h_ in h_layer_lst:
                        h_.set_H_container(dropout=sampled_dropout,
                                           weight_decay=sampled_weight_decay)
                        h_tensor_matrix.append(h_.to_torch_tensor())
                    h_matrix = torch.stack(h_tensor_matrix, dim=0).to(torch.float32).to(self.device)

                    W_batch = self.hnet.forward(cond_input=h_matrix)

                    for W_idx, W_weights in enumerate(W_batch):
                        ret_W_weights = self.architecture_masking(W_weights,
                                                                  amask=arch_matrix[W_idx],
                                                                  hidden_lst=hidden_list[used_l_num][W_idx])
                        pred = self.ae_model_lst[used_l_num].forward(X,
                                                                     weights=ret_W_weights,
                                                                     h_container=h_layer_lst[W_idx])[0]
                        individual_loss = criterion(pred, X)
                        loss = loss + individual_loss
                loss.backward()
                optimizer.step()
                loss_lst.append(loss.item())

                if verbose:
                    # individual losses
                    ind_loss_lst.append(self.compute_individual_loss(X,
                                                                     A_list,
                                                                     train_dropout,
                                                                     train_weight_decay,
                                                                     h_list,
                                                                     hidden_list,
                                                                     criterion,
                                                                     used_l_num))
                    # print information
            if epoch % 50 == 0 and verbose:
                print('[%d] loss: %.3f' % (epoch + 1, np.mean(loss_lst)))
            # save loss trajectories
            total_loss_trajectory.append(np.mean(loss_lst))

            if verbose:
                ind_loss_trajectory.append(ind_loss_lst)

            # evaluate during training: 
            # acquire AUROC, weight decay information with early stopping 
            if epoch % epoch_save_step == 0 and epoch > 0:
                used_num = len(A_list) - 1 - int(epoch / (self.train_epochs / len(A_list)))
                aurocs, pred_recon = self.evaluate(eval_loader=eval_loader,
                                                   eval_compression_lst=eval_compression_lst,
                                                   used_num=used_num,
                                                   eval_dropout=eval_dropout,
                                                   eval_weight_decay=eval_weight_decay)
                auroc_lst.append(aurocs)
                pred_recons_lst.append(pred_recon)

        return total_loss_trajectory, ind_loss_trajectory, auroc_lst, pred_recons_lst

    def evaluate(self,
                 eval_loader,
                 eval_compression_lst,
                 eval_dropout=[0.0, 0.2],
                 eval_weight_decay=[0.0, 1e-5],
                 used_num=0,
                 ):

        def remove_redundant(lst):
            old_lst = lst
            lst = [i for i in list(old_lst) if i != 0]
            if len(old_lst) > len(lst):
                return lst[0: int(len(lst) / 2)] + lst[int(len(lst) / 2 + 1):]
            else:
                return lst

        # find the architectures
        A_list, hidden_list, h_lst = self.build_architectures(eval_compression_lst)
        # all test labels
        test_label = []
        # all test index
        test_data_index = []

        # create a ret_dictionay
        ret_dict = {}
        ret_result_dict = {}
        ret_name_list = []
        for l_num in range(used_num, len(A_list)):
            for w_num in range(len(eval_compression_lst)):
                for ed in eval_dropout:
                    for wd in eval_weight_decay:
                        # print(len(hiddn_total_lst), len(hiddn_total_lst[0]), l_num, w_num)
                        ret_name = '[{}]'.format(', '.join(map(str, remove_redundant(hidden_list[l_num][w_num])))) \
                                   + " | " + str(ed) + " | " + str(wd)
                        if ret_name in ret_name_list:
                            continue
                        ret_name_list.append(ret_name)
                        ret_dict[ret_name] = 0.0
                        ret_result_dict[ret_name] = []

        # iterate through all possible architectures
        # find the ones associated with the dropout and weight decay
        # and acquire the prediction results
        for idx, data in enumerate(eval_loader):
            X = data[0].to(self.device)
            label = data[1]
            data_idx = data[2].detach().cpu().numpy()
            # append to test_data_index
            test_data_index.append(data_idx)
            ret_name_lst = []

            for l_num, arch_matrix in enumerate(A_list):
                # skip unused architectures
                # they are not propoerly trained with our scheduler
                if l_num < used_num:
                    continue

                for dropout_ in eval_dropout:
                    for weight_decay_ in eval_weight_decay:

                        # create the corresponding h_matrix (batched form of h_container,
                        # but set the dropout and weight decay to corresponding ones)
                        h_layer_lst = h_lst[l_num]
                        h_tensor_matrix = []
                        for h_ in h_layer_lst:
                            h_.set_H_container(dropout=dropout_,
                                               weight_decay=weight_decay_)
                            h_tensor_matrix.append(h_.to_torch_tensor())
                        h_matrix = torch.stack(h_tensor_matrix, dim=0).to(torch.float32).to(self.device)

                        # find the weights
                        W_batch = self.hnet.forward(cond_input=h_matrix)

                        # find the prediction to underlying tasks
                        for w_num, W_weights in enumerate(W_batch):
                            ret_name = '[{}]'.format(', '.join(map(str, remove_redundant(hidden_list[l_num][w_num])))) \
                                       + " | " + str(dropout_) + " | " + str(weight_decay_)
                            if ret_name in ret_name_lst:
                                continue
                            ret_name_lst.append(ret_name)
                            ret_W_weights = self.architecture_masking(W_weights,
                                                                      amask=arch_matrix[w_num],
                                                                      hidden_lst=hidden_list[l_num][w_num])
                            pred = self.ae_model_lst[l_num].forward(X,
                                                                    weights=ret_W_weights,
                                                                    h_container=h_layer_lst[w_num])[0]
                            reconstruction_loss = np.mean(np.square((pred - X).detach().cpu().numpy()), axis=1)
                            ret_result_dict[ret_name].append(reconstruction_loss)

            test_label.append(label.detach().cpu().numpy())
        test_data_index = np.concatenate(test_data_index, axis=0)
        cat_label = np.concatenate(test_label, axis=0)[test_data_index]

        # calculate per architecture auroc
        for ret_name in ret_result_dict.keys():
            ret_result_dict[ret_name] = np.concatenate(ret_result_dict[ret_name], axis=0)[test_data_index]
            auroc = roc_auc_score(cat_label, ret_result_dict[ret_name])
            ret_dict[ret_name] = auroc
        return ret_dict, ret_result_dict
