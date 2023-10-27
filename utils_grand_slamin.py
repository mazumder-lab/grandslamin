#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
#%%
import torch
from typing import Any, Optional
from torch import Tensor, nn
from torch.nn.parameter import Parameter
import copy
import math
from utils_inits_pytorch_geom import *
import optuna
import itertools
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, LinearLR, CosineAnnealingLR, LambdaLR
import time
import json

# https://www.kaggle.com/competitions/cs419m/data

# --------------------------
# --- Model Grand Slamin ---
# --------------------------

import torch
from torch.utils.data import DataLoader

class Grand_slamin(torch.nn.Module):
      def __init__(self, n_trees: int, n_main: int, n_interactions: int, depth: int, 
                  bias: bool = True, activation_func: Optional[str] = None, p_output: int = 1, 
                  n_features_kept: int = -1,
                  weight_initializer: Optional[str] = None,
                  bias_initializer: Optional[str] = None,
                  hierarchy: int = 0,
                  gamma: float = 1.0,
                  alpha: float = 1.0,
                  selection_reg: float = 0.1,
                  entropy_reg: float = 0.001,
                  l2_reg: float = 0.0001,
                  group_l1_reg: float = 0.0,
                  test_different_lr: int = 0,
                  steps_per_epoch: int=1,
                  dense_to_sparse: int=1,
                  type_of_task: str="classification",
                  sel_penalization_in_hierarchy: int=0,
                  val_second_lr: float=-1,
                  seed = -1,
                  device = "cpu",
                  meta_info = None):
            
            super().__init__()
            
            self.n_trees = n_trees
            self.p_output = p_output
            if p_output==2:
                  self.p_output = 1
            self.meta_info = meta_info
            self.depth = depth
            self.activation_func = activation_func
            self.bias = bias
            self.n_features_kept = n_features_kept
            self.type_of_task = type_of_task
            self.seed = seed
            self.val_second_lr = val_second_lr
            
            self.hierarchy = hierarchy
            self.gamma = gamma
            self.alpha = alpha
            self.selection_reg = selection_reg
            self.entropy_reg = entropy_reg
            self.l2_reg = l2_reg
            self.group_l1_reg = group_l1_reg
            self.test_different_lr = test_different_lr
            self.steps_per_epoch = steps_per_epoch
            self.dense_to_sparse = dense_to_sparse
            self.sel_penalization_in_hierarchy = sel_penalization_in_hierarchy

            self.save_output_models = False
            self.change_output = True

            self.n_main = n_main
            self.n_interactions = n_interactions
            self.weight_initializer = weight_initializer
            self.bias_initializer = bias_initializer

            self.deleted_main = []
            self.deleted_times_main = []
            self.deleted_inter = []
            self.deleted_times_inter = []
            self.iter_batch = 0
            
            if self.type_of_task == "regression":
                  self.has_cat = False
            else:
                  self.has_cat = "categorical" in [meta_info[x]["type"] for x in meta_info]
            if self.has_cat:
                  self.embeddings = [meta_info[x]["encoder"] for x in meta_info if "encoder" in meta_info[x]]
                  try:
                        self.embeddings = torch.nn.ModuleList(self.embeddings)
                  except:
                        pass
            if self.seed!=-1:
                  torch.random.manual_seed(self.seed)
            
            if self.n_features_kept!=-1 and self.has_cat:
                  print("ERRROR: n_features_kept!=-1 USED WITH CAT FEATURES, not implemented yet")
                  
            if not(self.has_cat):
                  self.weight_proba_main = Parameter(torch.Tensor(self.n_main, self.n_trees, 2**self.depth-1, 1))
            self.weight_output_main = Parameter(torch.Tensor(self.n_main, self.p_output, self.n_trees, 2**self.depth))
            if not(self.has_cat):
                  self.weight_proba_inter = Parameter(torch.Tensor(self.n_interactions, self.n_trees, 2**self.depth-1, 2))
            self.weight_output_inter = Parameter(torch.Tensor(self.n_interactions, self.p_output, self.n_trees, 2**self.depth))
            if self.bias:
                  self.bias_proba_main = Parameter(torch.Tensor(self.n_main, self.n_trees, 2**self.depth-1))
                  self.bias_proba_inter = Parameter(torch.Tensor(self.n_interactions, self.n_trees, 2**self.depth-1))
                  self.bias_output = Parameter(torch.Tensor(self.p_output))
            else:
                  self.register_parameter('bias_proba_main', None)
                  self.register_parameter('bias_proba_inter', None)
                  self.register_parameter('bias_output', None)

            self.weight_z_main = Parameter(torch.Tensor(self.n_main))
            self.weight_z_inter = Parameter(torch.Tensor(self.n_interactions))

            #self._load_hook = self._register_load_state_dict_pre_hook(
            #      self._lazy_load_hook)

            path_to_leaf_bool_numpy = (((np.arange(2**self.depth)[:,None] & (1 << np.arange(self.depth+1)))) > 0).astype(int).T
            path_to_leaf_bool_numpy = np.flip(path_to_leaf_bool_numpy,0)

            path_to_leaf_idx_numpy = np.copy(path_to_leaf_bool_numpy)
            for i in range(1,self.depth):
                  path_to_leaf_idx_numpy[i:,:]+=(2**i)*path_to_leaf_bool_numpy[:-i,:]
            path_to_leaf_idx_numpy+=2**np.arange(self.depth+1)[:,np.newaxis]
            path_to_leaf_idx_numpy=np.array(path_to_leaf_idx_numpy, dtype = int)
            idx_access = path_to_leaf_idx_numpy[:-1]-1
            bool_left = 1-path_to_leaf_bool_numpy[1:,:]
            self.idx_access = idx_access
            self.bool_left = torch.Tensor(bool_left).to(device)

            if self.n_features_kept!=-1 and not(self.has_cat):
                  self.indices_features_main = torch.argsort(torch.rand(*self.weight_proba_main.data.shape), dim=-1)
                  self.mask_features_main = torch.zeros(self.weight_proba_main.data.shape)
                  self.mask_features_main.scatter_(-1, self.indices_features_main[:,:,:,:self.n_features_kept], 1)
                  self.indices_features_inter = torch.argsort(torch.rand(*self.weight_proba_inter.data.shape), dim=-1)
                  self.mask_features_inter = torch.zeros(self.weight_proba_inter.data.shape)
                  self.mask_features_inter.scatter_(-1, self.indices_features_inter[:,:,:,:self.n_features_kept], 1)
                  self.mask_features_main = Parameter(self.mask_features_main, requires_grad=False)
                  self.mask_features_inter = Parameter(self.mask_features_inter, requires_grad=False)
            
            self.l_main = Parameter(torch.arange(n_main), requires_grad=False)
            self.device = device

      def update_indice_param(self):
            self.indices_param_main = [[],[]]
            self.indices_param_inter = [[],[]]
            list_params = list(self.named_parameters())
            self.d_regular={}
            self.d_diff={}
            acc_1 = 0
            acc_2 = 0
            for i in range(len(list_params)):
                  if "embedding" in list_params[i][0]:
                        # self.indices_param_main[0].append(acc_1)
                        self.d_regular[acc_1] = list_params[i][0]
                        acc_1 += 1
                  if "main" in list_params[i][0]:
                        if self.test_different_lr and "weight_z_main" in list_params[i][0]:
                              self.indices_param_main[1].append(acc_2)
                              self.d_diff[acc_2] = list_params[i][0]
                              acc_2 += 1
                        else:
                              self.indices_param_main[0].append(acc_1)
                              self.d_regular[acc_1] = list_params[i][0]
                              acc_1 += 1
                  if "inter" in list_params[i][0] or "combination" in list_params[i][0]:
                        if self.test_different_lr and "weight_z_inter" in list_params[i][0]:
                              self.indices_param_inter[1].append(acc_2)
                              self.d_diff[acc_2] = list_params[i][0]
                              acc_2 += 1
                        else:
                              self.indices_param_inter[0].append(acc_1)
                              self.d_regular[acc_1] = list_params[i][0]
                              acc_1 += 1

      def perform_screening(self, dataset, max_interaction_number) -> float:
            (dataset_train, corres_y_train), (dataset_val, corres_y_val), (dataset_test, corres_y_test), meta_info = dataset
            p_features = dataset_train.shape[-1]
            l_combinations = np.array(list(itertools.combinations(np.arange(p_features), 2)))
            if max_interaction_number != -1:
                  try:
                        data_numpy_train = dataset_train.cpu()
                        y_numpy_train = corres_y_train.cpu()
                        data_numpy_val = dataset_val.cpu()
                        y_numpy_val = corres_y_val.cpu()
                  except:
                        data_numpy_train = dataset_train
                        y_numpy_train = corres_y_train
                        data_numpy_val = dataset_val
                        y_numpy_val = corres_y_val
                  data_numpy_train = data_numpy_train[0].numpy()
                  y_numpy_train = y_numpy_train[0].numpy()
                  data_numpy_val = data_numpy_val[0].numpy()
                  y_numpy_val = y_numpy_val[0].numpy()
                  if self.type_of_task=="regression":
                        l_mse = Parallel(n_jobs=16, verbose=1)(delayed(get_mse_combinations)(data_numpy_train, y_numpy_train, data_numpy_val, y_numpy_val, l_combinations, idx_combination, self.seed) for idx_combination in range(len(l_combinations)))
                        l_mse = np.array(l_mse)
                        l_combinations = l_combinations[np.argsort(l_mse)][:max_interaction_number]
                  elif self.type_of_task == "classification":
                        l_auc = Parallel(n_jobs=16, verbose=1)(delayed(get_auc_combinations)(data_numpy_train, y_numpy_train, data_numpy_val, y_numpy_val, l_combinations, idx_combination, self.seed) for idx_combination in range(len(l_combinations)))
                        l_auc = np.array(l_auc)
                        l_combinations = l_combinations[np.argsort(-l_auc)][:max_interaction_number]

            if self.has_cat:
                  self.l_n_cat = [self.meta_info[key_feat]["n_cat_out"] if "n_cat_out" in self.meta_info[key_feat] else 1 for key_feat in self.meta_info][:-1]
                  self.l_cumsum_cat = np.cumsum([0]+self.l_n_cat)
                  self.l_main_proba = []
                  self.l_inter_proba_weights = []
                  self.l_inter_proba_inputs = []
                  self.l_inter_proba_inputs2 = []
                  for i in range(len(self.l_n_cat)):
                        n_cat = self.l_n_cat[i]
                        self.l_main_proba += [i for j in range(n_cat)]
                  self.l_main_proba = Parameter(torch.Tensor(self.l_main_proba).long(), requires_grad=False)
                  self.l_main_proba_idx = torch.arange(len(self.l_main_proba))
                  for i in range(len(l_combinations)):
                        n_cat_1 = self.l_n_cat[l_combinations[i][0]]
                        n_cat_2 = self.l_n_cat[l_combinations[i][1]]
                        self.l_inter_proba_weights += [i for j in range(n_cat_1+n_cat_2)]
                        self.l_inter_proba_inputs += [l_combinations[i][0] for j in range(n_cat_1)]
                        self.l_inter_proba_inputs += [l_combinations[i][1] for j in range(n_cat_2)]
                        self.l_inter_proba_inputs2 += [self.l_cumsum_cat[l_combinations[i][0]]+j for j in range(n_cat_1)]
                        self.l_inter_proba_inputs2 += [self.l_cumsum_cat[l_combinations[i][1]]+j for j in range(n_cat_2)]
                  self.l_inter_proba_weights = Parameter(torch.Tensor(self.l_inter_proba_weights).long(), requires_grad=False)
                  self.l_inter_proba_inputs = Parameter(torch.Tensor(self.l_inter_proba_inputs).long(), requires_grad=False)
                  self.l_inter_proba_inputs2 = Parameter(torch.Tensor(self.l_inter_proba_inputs2).long(), requires_grad=False)
                  self.n_main_proba = len(self.l_main_proba)
                  self.n_inter_proba = len(self.l_inter_proba_weights)
                  self.weight_proba_main = Parameter(torch.Tensor(self.n_main_proba, self.n_trees, 2**self.depth-1, 1))
                  self.weight_proba_inter = Parameter(torch.Tensor(self.n_inter_proba, self.n_trees, 2**self.depth-1, 1))

                  self.batch_norm_cat = torch.nn.BatchNorm1d(self.n_main_proba)

            dataset_train_main = dataset_train.permute((2,1,0))
            dataset_val_main = dataset_val.permute((2,1,0))
            dataset_test_main = dataset_test.permute((2,1,0))
            if self.has_cat:
                  dataset_train_inter = torch.zeros(0,dataset_train_main.shape[1], dataset_train_main.shape[2]) #dataset_train_main[l_combinations.flatten(),:,:]
                  dataset_val_inter = torch.zeros(0,dataset_val_main.shape[1], dataset_val_main.shape[2]) #dataset_val_main[l_combinations.flatten(),:,:]
                  dataset_test_inter = torch.zeros(0,dataset_test_main.shape[1], dataset_test_main.shape[2]) #dataset_test_main[l_combinations.flatten(),:,:]
            else:
                  dataset_train_inter = dataset_train[0,:,l_combinations].permute((1,0,2))
                  dataset_val_inter = dataset_val[0,:,l_combinations].permute((1,0,2))
                  dataset_test_inter = dataset_test[0,:,l_combinations].permute((1,0,2))
            corres_y_train = corres_y_train[0]
            corres_y_val = corres_y_val[0]
            corres_y_test = corres_y_test[0]
            
            try:
                  self.l_combinations = Parameter(torch.Tensor(l_combinations).long(), requires_grad=False)
                  self.l_combinations_idx = Parameter(torch.arange(l_combinations.shape[0]), requires_grad=False)
            except:
                  pass

            self.update_indice_param()
            self.reset_parameters_soft_trees()
            return dataset_train_main, dataset_val_main, dataset_test_main, dataset_train_inter, dataset_val_inter, dataset_test_inter, corres_y_train, corres_y_val, corres_y_test, meta_info

      def __deepcopy__(self, memo):
            out = Grand_slamin(
                  n_trees=self.n_trees,
                  n_main=self.n_main, 
                  n_interactions=self.n_interactions, 
                  depth=self.depth, 
                  bias=self.bias, 
                  activation_func=self.activation_func, 
                  p_output=self.p_output, 
                  n_features_kept=self.n_features_kept,
                  weight_initializer=self.weight_initializer,
                  bias_initializer=self.bias_initializer,
                  hierarchy=self.hierarchy,
                  gamma=self.gamma,
                  alpha=self.alpha,
                  selection_reg=self.selection_reg,
                  entropy_reg=self.entropy_reg,
                  l2_reg=self.l2_reg,
                  group_l1_reg=self.group_l1_reg,
                  test_different_lr=self.test_different_lr,
                  steps_per_epoch=self.steps_per_epoch,
                  dense_to_sparse=self.dense_to_sparse,
                  type_of_task=self.type_of_task,
                  sel_penalization_in_hierarchy=self.sel_penalization_in_hierarchy,
                  val_second_lr=self.val_second_lr,
                  seed = self.seed,
                  device = self.device,
                  meta_info=self.meta_info)
            
            out.weight_proba_main = copy.deepcopy(self.weight_proba_main, memo)
            out.weight_output_main = copy.deepcopy(self.weight_output_main, memo)
            out.weight_proba_inter = copy.deepcopy(self.weight_proba_inter, memo)
            out.weight_output_inter = copy.deepcopy(self.weight_output_inter, memo)

            out.weight_z_main = copy.deepcopy(self.weight_z_main, memo)
            out.weight_z_inter = copy.deepcopy(self.weight_z_inter, memo)

            if self.bias:
                  out.bias_proba_main = copy.deepcopy(self.bias_proba_main, memo)
                  out.bias_proba_inter = copy.deepcopy(self.bias_proba_inter, memo)
                  out.bias_output = copy.deepcopy(self.bias_output, memo)
            
            if self.n_features_kept!=-1:
                  out.mask_features_main = copy.deepcopy(self.mask_features_main, memo)
                  out.mask_features_inter = copy.deepcopy(self.mask_features_inter, memo)

            out.l_main = copy.deepcopy(self.l_main, memo)
            out.l_combinations = copy.deepcopy(self.l_combinations, memo)
            out.l_combinations_idx = copy.deepcopy(self.l_combinations_idx, memo)
            if self.has_cat:
                  out.l_main_proba = copy.deepcopy(self.l_main_proba, memo)
                  out.l_main_proba_idx = copy.deepcopy(self.l_main_proba_idx, memo)
                  out.l_inter_proba_weights = copy.deepcopy(self.l_inter_proba_weights, memo)
                  out.l_inter_proba_inputs = copy.deepcopy(self.l_inter_proba_inputs, memo)
                  out.l_inter_proba_inputs2 = copy.deepcopy(self.l_inter_proba_inputs2, memo)
            out.update_indice_param()
            out.deleted_main = copy.deepcopy(self.deleted_main, memo)
            out.deleted_inter = copy.deepcopy(self.deleted_inter, memo)
            out.deleted_times_main = copy.deepcopy(self.deleted_times_main, memo)
            out.deleted_times_inter = copy.deepcopy(self.deleted_times_inter, memo)
            if self.has_cat:
                  out.embeddings = copy.deepcopy(self.embeddings)
                  out.batch_norm_cat = copy.deepcopy(self.batch_norm_cat)
                  try:
                        out.embeddings = torch.nn.ModuleList(out.embeddings)
                  except:
                        pass
            return out

      def reset_parameters_soft_trees(self):
            if self.weight_initializer== "glorot":
                  glorot(self.weight_proba_main)
                  glorot(self.weight_output_main)
                  glorot(self.weight_proba_inter)
                  glorot(self.weight_output_inter)
            elif self.weight_initializer == 'uniform':
                  bound = 1.0 / math.sqrt(self.weight_proba_main.size(-1))
                  torch.nn.init.uniform_(self.weight_proba_main.data, -bound, bound)
                  bound = 1.0 / math.sqrt(self.weight_output_main.size(-1))
                  torch.nn.init.uniform_(self.weight_output_main.data, -bound, bound)
                  bound = 1.0 / math.sqrt(self.weight_proba_inter.size(-1))
                  torch.nn.init.uniform_(self.weight_proba_inter.data, -bound, bound)
                  bound = 1.0 / math.sqrt(self.weight_output_inter.size(-1))
                  torch.nn.init.uniform_(self.weight_output_inter.data, -bound, bound)
            elif self.weight_initializer == 'kaiming_uniform':
                  kaiming_uniform(self.weight_proba_main, fan=1,
                                    a=math.sqrt(5))
                  kaiming_uniform(self.weight_output_main, fan=1,
                                    a=math.sqrt(5))
                  kaiming_uniform(self.weight_proba_inter, fan=2,
                                    a=math.sqrt(5))
                  kaiming_uniform(self.weight_output_inter, fan=2,
                                    a=math.sqrt(5))
            elif self.weight_initializer is None:
                  kaiming_uniform(self.weight_proba_main, fan=1,
                                    a=math.sqrt(5))
                  kaiming_uniform(self.weight_output_main, fan=1,
                                    a=math.sqrt(5))
                  kaiming_uniform(self.weight_proba_inter, fan=2,
                                    a=math.sqrt(5))
                  kaiming_uniform(self.weight_output_inter, fan=2,
                                    a=math.sqrt(5))
            else:
                  raise RuntimeError(f"Linear layer weight initializer "
                                    f"'{self.weight_initializer}' is not supported")

            self.weight_z_main.data.uniform_(-self.gamma/100, self.gamma/100)
            self.weight_z_inter.data.uniform_(-self.gamma/100, self.gamma/100)

            if not(self.bias):
                  pass
            elif self.bias_initializer == 'zeros':
                  zeros(self.bias_proba_main)
                  zeros(self.bias_proba_inter)
                  zeros(self.bias_output)
            elif self.bias_initializer is None:
                  uniform(1, self.bias_proba_main)
                  uniform(2, self.bias_proba_inter)
                  uniform(1, self.bias_output)

            else:
                  raise RuntimeError(f"Linear layer bias initializer "
                                    f"'{self.bias_initializer}' is not supported")

      def get_losses(self) -> float:
            if self.entropy_reg!=0:
                  entropy_loss_main = -(self.entropy_reg * torch.sum(self.z_main*torch.log(self.z_main+1e-6)+(1-self.z_main)*torch.log(1-self.z_main+1e-6)))
                  entropy_loss_inter = -(self.entropy_reg * torch.sum(self.z_inter*torch.log(self.z_inter+1e-6)+(1-self.z_inter)*torch.log(1-self.z_inter+1e-6)))
            else:
                  entropy_loss_main = torch.tensor(0)
                  entropy_loss_inter = torch.tensor(0)
                  
            if self.selection_reg !=0:
                  selection_loss_main = self.selection_reg * torch.sum(self.z_main)
                  selection_loss_inter = self.selection_reg * torch.sum(self.z_inter)
            else:
                  selection_loss_main = torch.tensor(0)
                  selection_loss_inter = torch.tensor(0)

            if self.l2_reg !=0:
                  l2_loss_main = self.l2_reg * torch.sum(self.weight_output_main**2)
                  l2_loss_inter = self.l2_reg * torch.sum(self.weight_output_inter**2)
            else:
                  l2_loss_main = torch.tensor(0)
                  l2_loss_inter = torch.tensor(0)
            
            if self.group_l1_reg !=0:
                  group_l1_loss_main = torch.sum(self.weight_output_main**2, dim = [i for i in range(1,len(self.weight_output_main.shape))])
                  group_l1_loss_inter = torch.sum(self.weight_output_inter**2, dim=[i for i in range(1, len(self.weight_output_inter.shape))])
                  group_l1_loss_main = self.group_l1_reg * torch.sum(torch.sqrt(group_l1_loss_main))
                  group_l1_loss_inter = self.group_l1_reg * torch.sum(torch.sqrt(group_l1_loss_inter))
            else:
                  group_l1_loss_main = torch.tensor(0)
                  group_l1_loss_inter = torch.tensor(0)

            entropy_loss = entropy_loss_main + entropy_loss_inter
            selection_loss = selection_loss_main + self.alpha * selection_loss_inter
            l2_loss = l2_loss_main + l2_loss_inter
            group_l1_loss = group_l1_loss_main + group_l1_loss_inter
            return entropy_loss, selection_loss, l2_loss, group_l1_loss

      def get_n_z(self):
            if self.dense_to_sparse:
                  n_z_i = self.l_main.shape[0]
                  n_z_ij = self.l_combinations.shape[0]
                  all_features = torch.concat([self.l_main,self.l_combinations.flatten()])
                  try:
                        all_features = all_features.cpu()
                  except:
                        pass
                  n_features_used = torch.unique(all_features).shape[0]
            else:
                  idx_main_features = self.z_main.detach()
                  l_features_main = self.l_main[idx_main_features!=0]
                  idx_inter_features = self.q_inter.detach()
                  l_features_inter = self.l_combinations[idx_inter_features!=0]
                  l_features_used = torch.unique(torch.concat([l_features_main, l_features_inter.flatten()]))
                  n_z_i = len(l_features_main)
                  n_z_ij = len(l_features_inter)
                  n_features_used = len(l_features_used)
            return n_z_i, n_z_ij, n_features_used
      
      def prune_models(self, optimizer):
            test_pruned = False
            test_pruned_main = False
            test_pruned_inter = False
            with torch.no_grad():
                  if self.n_main>0:
                        condition_to_prune_main = self.z_main.detach()
                        if torch.min(condition_to_prune_main) == 0:
                              idx_keep_main = torch.where(condition_to_prune_main!=0)[0]
                              new_deleted = list(self.l_main[torch.where(condition_to_prune_main==0)[0]].cpu().numpy())
                              self.deleted_main+=new_deleted
                              self.deleted_times_main+=[self.iter_batch for i in range(len(new_deleted))]
                              copy_grad = copy.deepcopy(self.weight_proba_main.grad[idx_keep_main])
                              self.weight_proba_main = Parameter(self.weight_proba_main[idx_keep_main])
                              self.weight_proba_main.grad = copy_grad
                              copy_grad = copy.deepcopy(self.weight_output_main.grad[idx_keep_main])
                              self.weight_output_main = Parameter(self.weight_output_main[idx_keep_main])
                              self.weight_output_main.grad = copy_grad
                              copy_grad = copy.deepcopy(self.bias_proba_main.grad[idx_keep_main])
                              self.bias_proba_main = Parameter(self.bias_proba_main[idx_keep_main])
                              self.bias_proba_main.grad = copy_grad
                              copy_grad = copy.deepcopy(self.weight_z_main.grad[idx_keep_main])
                              self.weight_z_main = Parameter(self.weight_z_main[idx_keep_main])
                              self.weight_z_main.grad = copy_grad

                              self.l_main = Parameter(self.l_main[idx_keep_main], requires_grad=False)
                              self.n_main = len(idx_keep_main)
                              
                              test_pruned = True
                              test_pruned_main = True

                  if self.n_interactions>0:
                        condition_to_prune_inter = self.q_inter.detach()
                        if torch.min(condition_to_prune_inter) == 0:
                              idx_keep_inter = torch.where(condition_to_prune_inter!=0)[0]
                              new_deleted = [tuple(x.cpu().numpy()) for x in self.l_combinations[torch.where(condition_to_prune_inter==0)[0]]]
                              self.deleted_inter+=new_deleted
                              self.deleted_times_inter+=[self.iter_batch for i in range(len(new_deleted))]
                              # print("Deleted inter:",new_deleted, [self.iter_batch for i in range(len(new_deleted))])
                              copy_grad = copy.deepcopy(self.weight_proba_inter.grad[idx_keep_inter])
                              self.weight_proba_inter = Parameter(self.weight_proba_inter[idx_keep_inter])
                              self.weight_proba_inter.grad = copy_grad
                              copy_grad = copy.deepcopy(self.weight_output_inter.grad[idx_keep_inter])
                              self.weight_output_inter = Parameter(self.weight_output_inter[idx_keep_inter])
                              self.weight_output_inter.grad = copy_grad
                              copy_grad = copy.deepcopy(self.bias_proba_inter.grad[idx_keep_inter])
                              self.bias_proba_inter = Parameter(self.bias_proba_inter[idx_keep_inter])
                              self.bias_proba_inter.grad = copy_grad
                              copy_grad = copy.deepcopy(self.weight_z_inter.grad[idx_keep_inter])
                              self.weight_z_inter = Parameter(self.weight_z_inter[idx_keep_inter])
                              self.weight_z_inter.grad = copy_grad
                              self.l_combinations = Parameter(self.l_combinations[idx_keep_inter], requires_grad=False)
                              self.l_combinations_idx = Parameter(self.l_combinations_idx[idx_keep_inter], requires_grad=False)
                              self.n_interactions = len(idx_keep_inter)

                              test_pruned = True
                              test_pruned_inter = True
            if test_pruned:
                  self.update_indice_param()
                  optimizer_name = optimizer.__class__.__name__
                  copy_optimizer = initialize_optimizer(self.test_different_lr, self, optimizer_name, self.steps_per_epoch, optimizer.defaults["lr"], self.val_second_lr)
                  try:
                        copy_optimizer._step_count = optimizer._step_count
                  except:
                        pass
                  for idx_param in range(len(optimizer.param_groups)):
                        for key in optimizer.param_groups[idx_param]:
                              if key!="params":
                                    copy_optimizer.param_groups[idx_param][key] = optimizer.param_groups[idx_param][key]
                        list_params_old = list(optimizer.param_groups[idx_param]["params"])
                        list_params_new = list(copy_optimizer.param_groups[idx_param]["params"])
                        for i in range(len(list_params_new)):
                              if i in self.indices_param_main[idx_param]:
                                    group_param_old = list_params_old[i]
                                    group_param_new = list_params_new[i]
                                    copy_optimizer.state[group_param_new] = copy.deepcopy(optimizer.state[group_param_old])
                                    if test_pruned_main and optimizer_name == "Adam":
                                          if "exp_avg" in optimizer.state[group_param_old]:
                                                copy_optimizer.state[group_param_new]["exp_avg"] = copy.deepcopy(optimizer.state[group_param_old]["exp_avg"][idx_keep_main])
                                          if "exp_avg_sq" in optimizer.state[group_param_old]:
                                                copy_optimizer.state[group_param_new]["exp_avg_sq"] = copy.deepcopy(optimizer.state[group_param_old]["exp_avg_sq"][idx_keep_main])
                              elif i in self.indices_param_inter[idx_param]:
                                    group_param_old = list_params_old[i]
                                    group_param_new = list_params_new[i]
                                    copy_optimizer.state[group_param_new] = copy.deepcopy(optimizer.state[group_param_old])
                                    if test_pruned_inter and optimizer_name == "Adam":
                                          if "exp_avg" in optimizer.state[group_param_old]:
                                                copy_optimizer.state[group_param_new]["exp_avg"] = copy.deepcopy(optimizer.state[group_param_old]["exp_avg"][idx_keep_inter])
                                          if "exp_avg_sq" in optimizer.state[group_param_old]:
                                                copy_optimizer.state[group_param_new]["exp_avg_sq"] = copy.deepcopy(optimizer.state[group_param_old]["exp_avg_sq"][idx_keep_inter])
                              else:
                                    group_param_old = list_params_old[i]
                                    group_param_new = list_params_new[i]
                                    copy_optimizer.state[group_param_new] = copy.deepcopy(optimizer.state[group_param_old])
            else:
                  copy_optimizer = optimizer
            return copy_optimizer, test_pruned

      def prune_models_temp(self, optimizer):
            with torch.no_grad():
                  l_inf_weight_main = torch.abs(self.weight_output_main).norm(p=float("inf"), dim=[i for i in range(1,len(self.weight_output_main.shape))])
                  l_inf_weight_inter = torch.abs(self.weight_output_inter).norm(p=float("inf"), dim=[i for i in range(1,len(self.weight_output_inter.shape))])
                  condition_to_prune_main = self.z_main.detach()
                  condition_to_prune_inter = self.q_inter.detach()
                  if self.hierarchy==0 and torch.min(condition_to_prune_main) == 0:
                        idx_remove_main = torch.where(condition_to_prune_main==0)[0]
                        self.weight_z_main.data[idx_remove_main] = -self.gamma
                        self.z_main.data[idx_remove_main] = 0

                  if self.hierarchy==0 and torch.min(condition_to_prune_inter) == 0:
                        idx_remove_inter = torch.where(condition_to_prune_inter==0)[0]
                        self.weight_z_inter.data[idx_remove_inter] = -self.gamma
                        self.z_inter.data[idx_remove_inter] = 0

                  if self.hierarchy in [1,2] and torch.min(condition_to_prune_main) == 0:
                        idx_remove_main = torch.where(condition_to_prune_main==0)[0]
                        self.weight_z_main.data[idx_remove_main] = -self.gamma
                        self.z_main.data[idx_remove_main] = 0

            return optimizer, True

      def forward_proba(self, inputs) -> Tensor:
            print_cuda = False
            inputs_main, inputs_inter = inputs

            if print_cuda:
                  print("Cuda memory forward 1:", torch.cuda.memory_allocated("cuda"))
            
            if not(self.has_cat):
                  if self.n_features_kept!=-1:
                        first_step_main = torch.einsum("ijkl,iml -> ijkm", self.weight_proba_main*self.mask_features_main, inputs_main)
                        first_step_inter = torch.einsum("ijkl,iml -> ijkm", self.weight_proba_inter*self.mask_features_inter, inputs_inter)
                  else:
                        first_step_main = torch.einsum("ijkl,iml -> ijkm", self.weight_proba_main, inputs_main) # i=n_main, j=n_trees, k=n_nodes, l=p_output=1 here | m=n_samples
                        #first_step_main is (n_main, n_trees, n_nodes, n_samples)
                        first_step_inter = torch.einsum("ijkl,iml -> ijkm", self.weight_proba_inter, inputs_inter)
                        #first_step_inter is (n_inter, n_trees, n_nodes, n_samples)
            else:
                  first_step_main_temp = torch.einsum("ijkl,iml -> ijkm", self.weight_proba_main, inputs_main) # i=n_main, j=n_trees, k=n_nodes, l=p_output=1 here | m=n_samples
                  #first_step_main is (n_main_proba, n_trees, n_nodes, n_samples)
                  first_step_inter_temp = torch.einsum("ijkl,iml -> ijkm", self.weight_proba_inter, inputs_inter)
                  #first_step_inter is (n_inter_proba, n_trees, n_nodes, n_samples)
                  first_step_main = torch.zeros(tuple([self.n_main]+list(first_step_main_temp.shape[1:]))).to(first_step_main_temp)
                  first_step_inter = torch.zeros(tuple([self.n_interactions]+list(first_step_inter_temp.shape[1:]))).to(first_step_inter_temp)
                  first_step_main.index_add_(0, self.l_main_proba, first_step_main_temp)
                  first_step_inter.index_add_(0, self.l_inter_proba_weights, first_step_inter_temp)

            if print_cuda:
                  print("Cuda memory forward 2:", torch.cuda.memory_allocated("cuda"))
            if self.bias:
                  # biais_proba_main is (n_main, n_trees, n_nodes)
                  # biais_proba_inter is (n_inter, n_trees, n_nodes)
                  first_step_main.add_(self.bias_proba_main[:,:,:,None])
                  first_step_inter.add_(self.bias_proba_inter[:,:,:,None])
            if print_cuda:
                  print("Cuda memory forward 3:", torch.cuda.memory_allocated("cuda"))
            self.activation_func(first_step_main)
            probas_left_main = first_step_main #(n_main, n_trees, n_nodes, n_samples)
            self.activation_func(first_step_inter)
            probas_left_inter = first_step_inter #(n_inter, n_trees, n_nodes, n_samples)
            if print_cuda:
                  print("Cuda memory forward 4:", torch.cuda.memory_allocated("cuda"))
            proba_leaf_main = probas_left_main[:,:,self.idx_access]
            proba_leaf_main.mul_(2*self.bool_left[:,:,None]-1)
            proba_leaf_main.add_((1-self.bool_left[:,:,None]))
            #Same as: proba_leaf_main = probas_left_main[:,:,self.idx_access]*(self.bool_left[:,:,None]) + (1-probas_left_main[:,:,self.idx_access])*(1-self.bool_left[:,:,None])
            #or: proba_leaf_main = probas_left_main[:,:,self.idx_access]*(2*self.bool_left[:,:,None]-1) + (1-self.bool_left[:,:,None])
            if print_cuda:
                  print("Cuda memory forward 5:", torch.cuda.memory_allocated("cuda"))
            proba_leaf_inter = probas_left_inter[:,:,self.idx_access]
            if print_cuda:
                  print("Cuda memory forward 5 bis:", torch.cuda.memory_allocated("cuda"))
            proba_leaf_inter.mul_(2*self.bool_left[:,:,None]-1)
            if print_cuda:
                  print("Cuda memory forward 5 bis bis:", torch.cuda.memory_allocated("cuda"))
            proba_leaf_inter.add_((1-self.bool_left[:,:,None]))
            #Similar, same as: proba_leaf_inter = probas_left_inter[:,:,self.idx_access]*(self.bool_left[:,:,None]) + (1-probas_left_inter[:,:,self.idx_access])*(1-self.bool_left[:,:,None])
            if print_cuda:
                  print("Cuda memory forward 6:", torch.cuda.memory_allocated("cuda"))
            proba_leaf_main = torch.prod(proba_leaf_main, 2)
            proba_leaf_inter = torch.prod(proba_leaf_inter, 2)
            if print_cuda:
                  print("Cuda memory forward 7:", torch.cuda.memory_allocated("cuda"))
            return proba_leaf_main, proba_leaf_inter

      def forward(self, inputs) -> Tensor:
            print_cuda = False
            inputs_main, inputs_inter = inputs

            if self.change_output:
                  if self.has_cat:
                        inputs_main = inputs_main[:,:,0]
                        new_inputs_main = []
                        for ind in self.l_main:
                              ind = ind.item()
                              if self.meta_info["X"+str(ind+1)]["type"]=="categorical":
                                    new_inputs_main.append(self.embeddings[ind](inputs_main[ind].long()))
                              else:
                                    new_inputs_main.append(inputs_main[ind][:,None])
                        inputs_main = torch.hstack(new_inputs_main)
                        inputs_main.swapaxes_(0,1)
                        inputs_main = inputs_main[:,:,None]
                        inputs_inter = inputs_main[self.l_inter_proba_inputs2]
                  else:
                        inputs_main = inputs_main[self.l_main]
                        inputs_inter = inputs_inter[self.l_combinations_idx]

            proba_leaf_main, proba_leaf_inter = self.forward_proba((inputs_main, inputs_inter))
            # proba_leaf_main is (n_main, n_trees, n_leaves, n_samples)
            # self.weight_output_main is (n_main, p_output, n_trees, n_leaves)
            output_main = torch.einsum("ijkm,ipjk -> ipjkm", proba_leaf_main, self.weight_output_main)
            # output_main is (n_main, p_output, n_trees, n_leaves, n_samples)
      
            # proba_leaf_inter is (n_inter, n_trees, n_leaves, n_samples)
            # self.weight_output_inter is (n_inter, p_output, n_trees, n_leaves)
            output_inter = torch.einsum("ijkm,ipjk -> ipjkm", proba_leaf_inter, self.weight_output_inter)
            # output_inter is (n_inter, p_output, n_trees, n_leaves, n_samples)
            if print_cuda:
                  print("Cuda memory forward 8:", torch.cuda.memory_allocated("cuda"))
            del proba_leaf_inter, proba_leaf_main
            if print_cuda:
                  print("Cuda memory forward 9:", torch.cuda.memory_allocated("cuda"))
            output_main = torch.sum(output_main, 3)
            output_inter = torch.sum(output_inter, 3)
            # output_main is (n_main, p_output, n_trees, n_samples)
            # output_inter is (n_inter, p_output, n_trees, n_samples)
            if print_cuda:
                  print("Cuda memory forward 10:", torch.cuda.memory_allocated("cuda"))
            output_main = torch.sum(output_main, 2)
            output_inter = torch.sum(output_inter, 2)
            # output_main is (n_main, p_output, n_samples)
            # output_inter is (n_inter, p_output, n_samples)
            if print_cuda:
                  print("Cuda memory forward 11:", torch.cuda.memory_allocated("cuda"))
            condition_1_main = self.weight_z_main <= -self.gamma/2
            condition_2_main = self.weight_z_main >= self.gamma/2
            condition_1_inter = self.weight_z_inter <= -self.gamma/2
            condition_2_inter = self.weight_z_inter >= self.gamma/2

            smooth_zs_main = (-2 /(self.gamma**3)) * (self.weight_z_main**3) + (3/(2 * self.gamma)) * self.weight_z_main + 0.5
            smooth_zs_inter = (-2 /(self.gamma**3)) * (self.weight_z_inter**3) + (3/(2 * self.gamma)) * self.weight_z_inter + 0.5

            z_main = torch.where(condition_1_main, torch.zeros_like(self.weight_z_main), 
                              torch.where(condition_2_main, torch.ones_like(self.weight_z_main), smooth_zs_main))            
            if print_cuda:
                  print("Cuda memory forward 12:", torch.cuda.memory_allocated("cuda"))
            z_inter = torch.where(condition_1_inter, torch.zeros_like(self.weight_z_inter), 
                              torch.where(condition_2_inter, torch.ones_like(self.weight_z_inter), smooth_zs_inter))
            if self.hierarchy == 0:
                  q_inter = z_inter
            elif self.hierarchy==1:
                  if len(self.l_main)>0:
                        n_main_max = torch.max(self.l_main)+1
                  else:
                        n_main_max = 0
                  if len(self.l_combinations)>0:
                        n_inter_max = torch.max(self.l_combinations)+1
                  else:
                        n_inter_max = 0
                  z_main_copy = torch.zeros(max(n_main_max, n_inter_max)).to(z_main)
                  z_main_copy[self.l_main] = z_main
                  z_prod = torch.prod(z_main_copy[self.l_combinations], 1)
                  q_inter = z_inter*z_prod
            elif self.hierarchy==2:
                  if len(self.l_main)>0:
                        n_main_max = torch.max(self.l_main)+1
                  else:
                        n_main_max = 0
                  if len(self.l_combinations)>0:
                        n_inter_max = torch.max(self.l_combinations)+1
                  else:
                        n_inter_max = 0
                  z_main_copy = torch.zeros(max(n_main_max, n_inter_max)).to(z_main)
                  z_main_copy[self.l_main] = z_main
                  z_prod = torch.prod(z_main_copy[self.l_combinations], 1)
                  z_sum = torch.sum(z_main_copy[self.l_combinations], 1)
                  q_inter = z_inter*(z_sum-z_prod)
            if print_cuda:
                  print("Cuda memory forward 13:", torch.cuda.memory_allocated("cuda"))

            if self.save_output_models:
                  self.output_main_models = copy.deepcopy(output_main.detach()*z_main[:,None,None].detach())
                  self.output_inter_models = copy.deepcopy(output_inter.detach()*q_inter[:,None,None].detach())

            output_main = torch.sum(output_main*z_main[:,None,None], 0)
            output_inter = torch.sum(output_inter*q_inter[:,None,None],0)
            output = (output_main + output_inter).swapaxes(0,1)

            self.z_main = z_main
            self.z_inter = z_inter
            self.q_inter = q_inter
            # output is (n_samples, p_output)

            if self.bias:
                  #self.bias_output is (p_output)
                  output.add_(self.bias_output[None,:])
            if print_cuda:
                  print("Cuda memory forward 14:", torch.cuda.memory_allocated("cuda"))
            
            if self.p_output==1:
                  output=output[:,0]
            
            if self.type_of_task=="regression":
                  return output
            elif self.type_of_task=="classification":
                  if self.p_output!=1:
                        return torch.nn.Softmax(1)(output)
                  else:
                        return torch.nn.Sigmoid()(output)

      def _save_to_state_dict(self, destination, prefix, keep_vars):
            destination[prefix + 'weight_proba_main'] = self.weight_proba_main.detach()
            destination[prefix + 'weight_proba_inter'] = self.weight_proba_inter.detach()
            destination[prefix + 'weight_output_main'] = self.weight_output_main.detach()
            destination[prefix + 'weight_output_inter'] = self.weight_output_inter.detach()
            destination[prefix + 'weight_z_main'] = self.weight_z_main.detach()
            destination[prefix + 'weight_z_inter'] = self.weight_z_inter.detach()
            if self.bias:
                  destination[prefix + 'bias_proba_main'] = self.bias_proba_main.detach()
                  destination[prefix + 'bias_proba_inter'] = self.bias_proba_inter.detach()
                  destination[prefix + 'bias_output'] = self.bias_output.detach()
            if self.n_features_kept != -1:
                  destination[prefix + 'mask_features_main'] = self.mask_features_main.detach()
                  destination[prefix + 'mask_features_inter'] = self.mask_features_inter.detach()
            destination[prefix + 'l_main'] = self.l_main
            destination[prefix + 'l_combinations'] = self.l_combinations
            destination[prefix + 'l_combinations_idx'] = self.l_combinations_idx

      def __repr__(self) -> str:
            return (f'{self.__class__.__name__}(n_trees={self.n_trees}, n_main={self.n_main}, n_interactions={self.n_interactions}, depth={self.depth}, n_features_kept={self.n_features_kept}, weight_initializer={self.weight_initializer}, bias_initializer={self.bias_initializer}, hierarchy={self.hierarchy}, gamma={self.gamma}, selection_reg={self.selection_reg}, entropy_reg={self.entropy_reg}, bias={self.bias})')

def read_model(path_study, ind_repeat, device, steps_per_epoch):
    path_model = path_study+"/best_trial/repeat_"+str(ind_repeat)+"/model"
    path_params = path_study+"/best_trial/params.json"

    with open(path_params, "r") as f:
        dict_params = json.load(f)

    if "group_l1_reg" not in dict_params:
        dict_params["group_l1_reg"] = 0.0
    if "val_second_lr" not in dict_params:
        dict_params["val_second_lr"] = -1

    weight_model = torch.load(path_model, map_location=device)
    l_main = weight_model["l_main"]
    l_combinations_idx = weight_model["l_combinations_idx"]
    n_main = len(l_main)
    n_interactions = len(l_combinations_idx)
    p_output = weight_model["weight_output_main"].shape[1]

    model = Grand_slamin(n_trees=dict_params["n_trees"], 
                              n_main=n_main, 
                              n_interactions=n_interactions, 
                              depth=dict_params["depth"], 
                              bias=True,
                              activation_func= tempered_sigmoid(dict_params["temperature"]),
                              p_output=p_output,
                              n_features_kept = dict_params["n_features_kept"],
                              weight_initializer = "glorot",
                              bias_initializer = "zeros",
                              hierarchy = dict_params["hierarchy"],
                              gamma = dict_params["gamma"],
                              alpha = dict_params["alpha"],
                              selection_reg = dict_params["selection_reg"],
                              entropy_reg = dict_params["entropy_reg"],
                              l2_reg=dict_params["l2_reg"],
                              group_l1_reg=dict_params["group_l1_reg"],
                              test_different_lr=dict_params["test_different_lr"],
                              steps_per_epoch=steps_per_epoch,
                              dense_to_sparse=dict_params["dense_to_sparse"],
                              type_of_task=dict_params["type_of_task"],
                              sel_penalization_in_hierarchy=dict_params["sel_penalization_in_hierarchy"],
                              val_second_lr=dict_params["val_second_lr"],
                              seed = dict_params["seed"]+ind_repeat,
                              device = device)    
    model.l_combinations = weight_model["l_combinations"]
    model.l_combinations_idx = weight_model["l_combinations_idx"]
    model.l_main = weight_model["l_main"]
    model.load_state_dict(weight_model)

    return model

# ----------------------------
# --- Dataset Grand Slamin ---
# ----------------------------

class Dataset_grand_slamin(torch.utils.data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, x_main, x_inter, y):
            'Initialization'
            self.x_main = x_main
            self.x_inter = x_inter
            self.y = y

      def __len__(self):
            'Denotes the total number of samples'
            return self.x_main.shape[0]

      def __getitem__(self, index):
            'Generates one sample of data'
            return (self.x_main[index], self.x_inter[index]), self.y[index]

# ---------------
# --- Metrics ---
# ---------------

def get_auc_combinations(data_numpy_train, y_numpy_train, data_numpy_val, y_numpy_val, l_combinations, idx_combination, seed):
    if seed==-1:
          model = DecisionTreeClassifier(max_depth=3)
    else:
          model = DecisionTreeClassifier(max_depth=3, random_state=seed)
          
    model.fit(data_numpy_train[:,l_combinations[idx_combination]], y_numpy_train)
    pred = model.predict_proba(data_numpy_val[:,l_combinations[idx_combination]])
    auc = compute_auc(y_numpy_val, pred)
    return auc

def get_mse_combinations(data_numpy_train, y_numpy_train, data_numpy_val, y_numpy_val, l_combinations, idx_combination, seed):
    if seed==-1:
          model = DecisionTreeRegressor(max_depth=3)
    else:
          model = DecisionTreeRegressor(max_depth=3, random_state=seed)
    model.fit(data_numpy_train[:,l_combinations[idx_combination]], y_numpy_train)
    pred = model.predict(data_numpy_val[:,l_combinations[idx_combination]])
    mse = np.mean((y_numpy_val-pred)**2)
    return mse

# -----------------
# --- Optimizer ---
# -----------------

def initialize_optimizer(test_different_lr, model, optimizer_name, steps_per_epoch, lr, val_second_lr):
      optimizer_func = getattr(torch.optim, optimizer_name)
      dict_params = dict(model.named_parameters())
      if test_different_lr:
            l_params_regular_lr = []
            l_params_modified_lr = []
            for i in range(len(model.d_regular)):
                  l_params_regular_lr.append(dict_params[model.d_regular[i]])
            for i in range(len(model.d_diff)):
                  l_params_modified_lr.append(dict_params[model.d_diff[i]])
            
            if val_second_lr== -1:
                  val_second_lr = lr/steps_per_epoch
            optimizer = optimizer_func([{"params":l_params_regular_lr}, {"params":l_params_modified_lr, "lr":val_second_lr}], lr = lr)
      else:
            l_params_regular_lr = []
            for i in range(len(model.d_regular)):
                  l_params_regular_lr.append(dict_params[model.d_regular[i]])
            optimizer = optimizer_func(l_params_regular_lr, lr = lr)
      return optimizer

# -----------------------------
# --- Training Grand Slamin ---
# -----------------------------

def tempered_sigmoid(temperature):
    def f(x):
        x.mul_(temperature)
        x.sigmoid_()
    return f

def train_grand_slamin(name_study, model, dataset, optimizer, criterion, n_epochs, batch_size_SGD, path_save, test_early_stopping, trial, type_decay="exponential", gamma_lr_decay=np.exp(-np.log(25)/10000), T_max_cos=10, eta_min_cos=1e-5, start_lr_decay=1e-2, end_lr_decay=1e-5, warmup_steps=100, type_of_task = "regression", test_compute_accurate_in_sample_loss = 0, folder_saves = "Saves_grand_slamin", ind_repeat=0, patience=50, metric_early_stopping="val_loss", period_milestones=25):
      if type_of_task=="regression":
            metric_name = "mse"
      elif type_of_task=="classification":
            metric_name = "accuracy"
      if type_decay!="None":
            print("type_decay diff from None", type_decay)
            if type_decay=="divergence":
                  scheduler = ExponentialLR(optimizer, gamma=gamma_lr_decay)
                  current_min_loss = np.inf
                  acc_divergence = 0
            if type_decay=="exponential":
                  scheduler = ExponentialLR(optimizer, gamma=gamma_lr_decay)
            if type_decay=="multi_lr":
                  n_milesones = n_epochs//period_milestones
                  milestones = period_milestones*np.arange(n_milesones)[1:]
                  scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma_lr_decay)
            if type_decay=="linear":
                  scheduler = LinearLR(optimizer, start_factor=start_lr_decay, end_factor=end_lr_decay, total_iters=n_epochs)
            if type_decay=="cosine":
                  scheduler = CosineAnnealingLR(optimizer, T_max=T_max_cos, eta_min=eta_min_cos)
            if type_decay=="ramp":
                  def warmup(current_step: int):
                        if current_step < warmup_steps:
                              # current_step / warmup_steps * base_lr
                              return float(current_step / warmup_steps)
                        else:
                              # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
                              return max(0.0, float(n_epochs - current_step) / float(max(1, n_epochs - warmup_steps)))
                  scheduler = LambdaLR(optimizer, lr_lambda=warmup)

      dataset_train_main, dataset_val_main, dataset_test_main, dataset_train_inter, dataset_val_inter, dataset_test_inter, corres_y_train, corres_y_val, corres_y_test, meta_info = dataset
      if "scaler" in meta_info["Y"]:
            scaler_y = meta_info["Y"]["scaler"]
      else:
            scaler_y = None

      l_lr = []
      l_in_sample_loss = []
      l_in_sample_metric = []
      l_validation_loss = []
      l_validation_metric = []
      l_n_z_i = []
      l_n_z_ij = []
      l_n_features_used = []
      l_times_epochs = []

      best_ep = 0
      best_model = copy.deepcopy(model)
      best_val_mse = np.inf
      best_train_loss = np.inf
      best_val_loss = np.inf
      best_val_acc = -np.inf

      n_epochs_no_improvement = 0
      generator = torch.Generator()
      if model.seed != -1:
            torch.random.manual_seed(model.seed)
            generator = generator.manual_seed(model.seed)
            
      loader_SGD = DataLoader(Dataset_grand_slamin(dataset_train_main.swapaxes(0,1), dataset_train_inter.swapaxes(0,1), corres_y_train), batch_size=batch_size_SGD, shuffle = True, generator=generator)
      
      for epoch in range(n_epochs):
            if (n_epochs_no_improvement < patience) or not(test_early_stopping):
                  start_epoch = time.time()
                  print("n_epochs_no_improvement =", n_epochs_no_improvement)
                  loss_pred_in_sample = 0
                  loss_pred_total = 0
                  model.train()
                  current_lr = optimizer.param_groups[0]["lr"]
                  n_seen = 0
                  for batch_sgd in loader_SGD:
                        n_batch = batch_sgd[0][0].shape[0]
                        n_seen += n_batch
                        optimizer.zero_grad()
                        output = model((batch_sgd[0][0].swapaxes(0,1), batch_sgd[0][1].swapaxes(0,1)))
                        if type_of_task == "regression":
                              loss = criterion(output, batch_sgd[1])
                        elif type_of_task == "classification":
                              loss = criterion(output, batch_sgd[1])
                        loss_pred_in_sample += n_batch*loss.detach().item()
                        entropy_loss, selection_loss, l2_loss, group_l1_loss = model.get_losses()
                        loss += entropy_loss + selection_loss + l2_loss + group_l1_loss
                        loss_pred_total += n_batch*loss.detach().item()
                        loss.backward()  # Derive gradients.
                        optimizer.step()  # Update parameters based on gradients.
                        if model.dense_to_sparse:
                              optimizer, test_pruned = model.prune_models(optimizer)
                              if type_decay=="multi_lr" and test_pruned:
                                    copy_scheduler = MultiStepLR(optimizer, milestones=scheduler.milestones, gamma=scheduler.gamma)
                                    copy_scheduler.load_state_dict(scheduler.state_dict())
                                    scheduler = copy_scheduler
                        else:
                              optimizer, test_pruned = model.prune_models_temp(optimizer)
                        model.iter_batch += 1
                  loss_in_sample = loss_pred_in_sample/n_seen
                  loss_pred_total = loss_pred_total/n_seen
                  try:
                        loss_in_sample = loss_in_sample.detach().item()
                  except:
                        pass
                  optimizer.zero_grad()
                  model.eval()
                  out = torch.zeros((0)).to(dataset_val_main.device)
                  with torch.no_grad():
                        size_batch = 100
                        n_batches = int(np.ceil(dataset_val_main.shape[1]/size_batch))
                        for ind_batch in range(n_batches):
                              idx_start = ind_batch*size_batch
                              idx_end = (ind_batch+1)*size_batch
                              out = torch.concat([out, model((dataset_val_main[:,idx_start:idx_end], dataset_val_inter[:,idx_start:idx_end]))])
                  if type_of_task == "regression":
                        mse_val = criterion(out.detach(), corres_y_val).item()
                        try:
                              mse_val = mse_val.cpu()
                        except:
                              pass
                        val_loss = mse_val
                  elif type_of_task == "classification":
                        if len(out.shape)>=2:
                              acc_val = torch.mean((torch.argmax(out.detach(), dim=1)==corres_y_val).float())*100
                        else:
                              acc_val = torch.mean((torch.round(out.detach())==corres_y_val).float())*100
                        try:
                              acc_val = acc_val.cpu()
                        except:
                              pass
                        val_loss = criterion(out.detach(), corres_y_val).item()
                  out = torch.zeros((0)).to(dataset_train_main.device)
                  with torch.no_grad():
                        size_batch = 100
                        n_batches = int(np.ceil(dataset_train_main.shape[1]/size_batch))
                        for ind_batch in range(n_batches):
                              idx_start = ind_batch*size_batch
                              idx_end = (ind_batch+1)*size_batch
                              out = torch.concat([out, model((dataset_train_main[:,idx_start:idx_end], dataset_train_inter[:,idx_start:idx_end]))])
                  if type_of_task == "regression":
                        mse_train = criterion(out.detach(), corres_y_train).item()
                        try:
                              mse_train = mse_train.cpu()
                        except:
                              pass
                        entropy_loss, selection_loss, l2_loss, group_l1_loss = model.get_losses()
                        mse_train += entropy_loss.detach().item() + selection_loss.detach().item() + l2_loss.detach().item() + group_l1_loss.detach().item()
                  elif type_of_task == "classification":
                        if len(out.shape)>=2:
                              acc_train = torch.mean((torch.argmax(out.detach(), dim=1)==corres_y_train).float())*100
                        else:
                              acc_train = torch.mean((torch.round(out.detach())==corres_y_train).float())*100
                        try:
                              acc_train = acc_train.cpu()
                        except:
                              pass
                  if epoch==0:
                        path_save_initial_model = folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat)+"/history"
                        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number))):
                              os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number))
                        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat))):
                              os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(trial.number)+"/repeat_"+str(ind_repeat))
                        if not(os.path.exists(path_save_initial_model)):
                              os.mkdir(path_save_initial_model)
                        torch.save(model.state_dict(), path_save_initial_model+"/model_"+str(epoch))

                  if test_compute_accurate_in_sample_loss or not(test_early_stopping):
                        model.eval()
                        out = model((dataset_train_main,dataset_train_inter))
                        entropy_loss, selection_loss, l2_loss, group_l1_loss = model.get_losses()
                        if type_of_task == "regression":
                              train_loss = criterion(out.detach(), corres_y_train).item()
                        elif type_of_task == "classification":
                              train_loss = criterion(out.detach(), corres_y_train).item()
                        train_loss += entropy_loss.detach().item() + selection_loss.detach().item() + l2_loss.detach().item() + group_l1_loss.detach().item()

                  if test_early_stopping:
                        if (type_of_task=="regression"):
                              if (mse_val < best_val_mse):
                                    best_val_mse = mse_val
                                    best_ep = epoch
                                    if path_save!=None:
                                          torch.save(model.state_dict(), path_save)
                                    best_model = copy.deepcopy(model)
                                    n_epochs_no_improvement = 0
                              else:
                                    n_epochs_no_improvement += 1
                        if (type_of_task=="classification"):
                              if metric_early_stopping == "val_loss":
                                    condition_improvement = (val_loss < best_val_loss)
                              elif metric_early_stopping == "val_accuracy":
                                    condition_improvement = (acc_val > best_val_acc)
                              if condition_improvement:
                                    best_val_loss = val_loss
                                    best_val_acc = acc_val
                                    best_ep = epoch
                                    if path_save!=None:
                                          torch.save(model.state_dict(), path_save)
                                    best_model = copy.deepcopy(model)
                                    n_epochs_no_improvement = 0
                              else:
                                    n_epochs_no_improvement += 1
                  else:
                        if train_loss < best_train_loss:
                              best_train_loss = train_loss
                              if type_of_task == "regression":
                                    best_val_mse = mse_val
                              elif type_of_task == "classification":
                                    best_val_acc = acc_val
                              best_ep = epoch
                              if path_save!=None:
                                    torch.save(model.state_dict(), path_save)
                              best_model = copy.deepcopy(model)

                  if test_compute_accurate_in_sample_loss or not(test_early_stopping):
                        print_loss = "In-sample Loss"
                        value_loss = train_loss
                  else:
                        print_loss = "Approx in-sample Loss"
                        value_loss = loss_in_sample
                  n_z_i, n_z_ij, n_features_used = model.get_n_z()
                  if type_of_task=="regression":
                        print(f'Epoch: {epoch:03d}, {print_loss}: {value_loss:.4f}, Approx total Loss: {loss_pred_total:.4f}, Validation loss: {val_loss:.4f}, Validation MSE: {mse_val:.4f}, lr: {current_lr:.4f}, n_z_i: {n_z_i:.4f}, n_z_ij: {n_z_ij:.4f}, n_features_used: {n_features_used:.4f}', flush=True)
                  elif type_of_task == "classification":
                        print(f'Epoch: {epoch:03d}, {print_loss}: {value_loss:.4f}, Approx total Loss: {loss_pred_total:.4f}, Validation loss: {val_loss:.4f}, Train Acc: {acc_train:.4f} Val Acc: {acc_val:.4f}, lr: {current_lr:.4f}, n_z_i: {n_z_i:.4f}, n_z_ij: {n_z_ij:.4f}, n_features_used: {n_features_used:.4f}', flush=True)
                  l_lr.append(current_lr)
                  l_in_sample_loss.append(value_loss)
                  l_validation_loss.append(val_loss)
                  l_n_z_i.append(n_z_i)
                  l_n_z_ij.append(n_z_ij)
                  l_n_features_used.append(n_features_used)
                  if type_of_task == "regression":
                        l_validation_metric.append(mse_val)
                        l_in_sample_metric.append(mse_train)
                  elif type_of_task == "classification":
                        l_validation_metric.append(acc_val)
                        l_in_sample_metric.append(acc_train)
                  l_times_epochs.append(time.time()-start_epoch)
                  if np.isnan(value_loss):
                        l_lr = l_lr + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_in_sample_loss = l_in_sample_loss + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_in_sample_metric = l_in_sample_metric + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_validation_loss = l_validation_loss + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_validation_metric = l_validation_metric + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_n_z_i = l_n_z_i + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_n_z_ij = l_n_z_ij + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_n_features_used = l_n_features_used + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        l_times_epochs = l_times_epochs + [np.nan for _ in range(max(n_epochs-epoch-1,0))]
                        break
                  if type_decay!="None":
                        scheduler.step()
                  if trial!=None:
                        if trial.should_prune():
                              raise optuna.exceptions.TrialPruned()
            else:
                  print("Early stopping at epoch", epoch)
                  break
      print("Delete loader")
      del loader_SGD

      l_in_sample_loss = np.array(l_in_sample_loss)
      l_validation_loss = np.array(l_validation_loss)
      l_in_sample_metric = np.array(l_in_sample_metric)
      l_validation_metric = np.array(l_validation_metric)
      l_times_epochs = np.array(l_times_epochs)
      l_lr = np.array(l_lr)
      l_n_z_i = np.array(l_n_z_i)
      l_n_z_ij = np.array(l_n_z_ij)
      l_n_features_used = np.array(l_n_features_used)
      in_sample_metric = evaluate_grand_slamin(best_model, (dataset_train_main, dataset_train_inter, corres_y_train), type_of_task, scaler_y=scaler_y)
      validation_metric = evaluate_grand_slamin(best_model, (dataset_val_main,dataset_val_inter,corres_y_val), type_of_task, scaler_y=scaler_y)
      test_metric = evaluate_grand_slamin(best_model, (dataset_test_main,dataset_test_inter,corres_y_test), type_of_task, scaler_y=scaler_y)

      try:
            in_sample_metric = in_sample_metric.cpu()
            validation_metric = validation_metric.cpu()
            test_metric = test_metric.cpu()
      except:
            pass
      
      try:
            in_sample_metric = in_sample_metric.item()
            validation_metric = validation_metric.item()
            test_metric = test_metric.item()
      except:
            pass

      dict_list = {}
      dict_list["l_in_sample_loss/In-sample loss"] = l_in_sample_loss
      dict_list["l_in_sample_metric/In-sample "+metric_name] = l_in_sample_metric
      dict_list["l_validation_loss/Validation loss"] = l_validation_loss
      dict_list["l_validation_metric/Validation "+metric_name] = l_validation_metric
      dict_list["l_times_epochs/Time per epoch"] = l_times_epochs
      dict_list["l_lr/Learning rate"] = l_lr
      dict_list["l_n_z_i/Number of z_i"] = l_n_z_i
      dict_list["l_n_z_ij/Number of z_ij"] = l_n_z_ij
      dict_list["l_n_features_used/Number of features used"] = l_n_features_used
      
      d_results = {}
      d_results["best_ep"] = best_ep
      if type_of_task == "regression":
            d_results["train_mse"] = in_sample_metric
            d_results["val_mse"] = validation_metric
            d_results["test_mse"] = test_metric
      elif type_of_task == "classification":
            d_results["train_acc"] = in_sample_metric[0]
            d_results["val_acc"] = validation_metric[0]
            d_results["test_acc"] = test_metric[0]  
            d_results["train_auc"] = in_sample_metric[1]
            d_results["val_auc"] = validation_metric[1]
            d_results["test_auc"] = test_metric[1]
      n_z_i, n_z_ij, n_features_used = best_model.get_n_z()
      d_results["n_z_i"] = n_z_i
      d_results["n_z_ij"] = n_z_ij
      d_results["n_features_used"] = n_features_used

      d_results["n_params"] = int(sum([np.prod(x.shape) for x in  list(best_model.parameters()) if x.requires_grad]))
      
      return d_results, validation_metric, best_model, dict_list

# ------------------
# --- Evaluation ---
# ------------------

def compute_auc(y_true, y_pred):
    if len(y_pred.shape)==2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:,1]
    auc = roc_auc_score(
        y_true,
        y_pred,
        multi_class="ovo"
    )
    return auc

def evaluate_grand_slamin(model, dataset, type_of_task, scaler_y=None):
      dataset_torch_main, dataset_torch_inter, corres_y = dataset
      model.eval()
      pred = torch.zeros((0)).to(dataset_torch_main.device)
      with torch.no_grad():
            size_batch = 100
            n_batches = int(np.ceil(dataset_torch_main.shape[1]/size_batch))
            for ind_batch in range(n_batches):
                  idx_start = ind_batch*size_batch
                  idx_end = (ind_batch+1)*size_batch
                  pred = torch.concat([pred, model((dataset_torch_main[:,idx_start:idx_end], dataset_torch_inter[:,idx_start:idx_end]))])
      if type_of_task=="regression":
            in_sample_pred = scaler_y.inverse_transform(pred.detach().cpu().numpy()[:,np.newaxis])[:,0]
            in_sample_truth = scaler_y.inverse_transform(corres_y.cpu().numpy()[:,np.newaxis])[:,0]
            metric = np.mean((in_sample_pred - in_sample_truth)**2)
      elif type_of_task=="classification":
            try:
                  pred_numpy = pred.cpu()
                  pred_numpy = pred_numpy.detach()
            except:
                  pass
            pred_numpy = pred_numpy.numpy()
            try:
                  corres_y_numpy = corres_y.cpu()
            except:
                  pass
            corres_y_numpy = corres_y_numpy.numpy()
            auc = compute_auc(corres_y_numpy, pred_numpy)
            if len(pred.shape)>=2:
                  acc = torch.mean((torch.argmax(pred.detach(), dim=1)==corres_y).float()).item()
            else:
                  acc = torch.mean((torch.round(pred.detach())==corres_y).float()).item()
            metric = (acc, auc)
      return metric

# ---------------------
# --- Visualization ---
# ---------------------

def purify_model(model, X_main, interaction_terms, active_interaction_terms):
    l_main = model.l_main
    l_combinations = model.l_combinations
    l_combinations_idx = model.l_combinations_idx
    n_main_original = len(X_main)

    model.save_output_models = True
    model.change_output = False

    fmain = {}
    acc = 0
    for i in range(n_main_original):
        if i in l_main:  
            x = torch.Tensor(-np.inf*np.ones((1, X_main[i].shape[0], n_main_original)))
            x[0,:,-1] = X_main[i]
            x_main = x.permute((2,1,0))
            x_inter = x[0,:,l_combinations].permute((1,0,2))
            model((x_main[l_main], x_inter))
            fmain[i] = model.output_main_models[acc,0,:].numpy()
            acc+=1
        else:
            fmain[i] = torch.zeros((X_main[i].shape[0])).numpy()

    X_interaction = {}
    finteraction = {}
    acc = 0
    for index, term in enumerate(interaction_terms):
        xixj = np.meshgrid(X_main[term[0]],X_main[term[1]],indexing='ij')
        X_interaction[term] = np.stack(xixj)
        xixj = np.vstack([xixj[0].reshape(-1),xixj[1].reshape(-1)]).T
        x = -np.inf*np.ones((xixj.shape[0],n_main_original))
        x[:,term[0]] = xixj[:,0]
        x[:,term[1]] = xixj[:,1]
        X_main_current = torch.Tensor(x.T[:,:,np.newaxis])
        X_inter_current = X_main_current.permute(2,1,0)[0,:,l_combinations].permute((1,0,2))
        model((X_main_current[l_main], X_inter_current))
        if index in l_combinations_idx:
            finteraction[term] = model.output_inter_models[acc,0,:].numpy().reshape(X_main[term[0]].shape[0],X_main[term[1]].shape[0])
            acc += 1
        else:
            finteraction[term] = np.zeros((X_main[term[0]].shape[0],X_main[term[1]].shape[0]))

    f_interaction_avg = {}

    for term in interaction_terms:

        f_interaction_avg[term] = {
            term[0]: finteraction[term].mean(axis=1, keepdims=True), term[1]: finteraction[term].mean(axis=0, keepdims=True)
        }

    for term in active_interaction_terms:
 
        while (np.abs(f_interaction_avg[term][term[0]])>1e-6).any() or (np.abs(f_interaction_avg[term][term[1]])>1e-6).any():
            for index, ij in enumerate(term):
                finteraction[term] -= f_interaction_avg[term][term[index]]
                fmain[term[index]] += f_interaction_avg[term][term[index]].reshape(-1)
                f_interaction_avg[term][term[index]] = finteraction[term].mean(axis=1, keepdims=True)
                f_interaction_avg[term][term[1-index]] = finteraction[term].mean(axis=0, keepdims=True)


    for i in l_main.numpy():
        if np.linalg.norm(fmain[i])>0:
            fmain[i] -= np.mean(fmain[i])
    return fmain, finteraction, X_main, X_interaction

print("Import utils grand slamin done!")