#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pandas as pd
import numpy as np
#%%
import torch
import optuna
from optuna.trial import TrialState
import logging
import sys
import matplotlib
matplotlib.use("Agg")
import shutil
import json
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import scipy.io
from sklearn.impute import SimpleImputer
import pickle
from sklearn.preprocessing import OrdinalEncoder

# -------------------------------
# --- Fine tuning with Optuna ---
# -------------------------------

class SaveStudyOptuna:
      def __init__(self, name_study, folder_saves):
            self.name_study = name_study
            self.folder_saves = folder_saves
            try:
                  if not(os.path.exists(self.folder_saves)):
                        os.mkdir(self.folder_saves)
            except:
                  pass
            if not(os.path.exists(self.folder_saves+"/study_"+self.name_study)):
                  os.mkdir(self.folder_saves+"/study_"+self.name_study)
      def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
            with open(self.folder_saves+"/study_"+self.name_study+"/save_study.pkl", 'wb') as handle:
                  pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
            if len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])) > 1:
                  param_imp_plot = optuna.visualization.plot_param_importances(study)
                  param_imp_plot.write_html(self.folder_saves+"/study_"+self.name_study+"/param_importance.html")
                  progress_over_time = optuna.visualization.plot_optimization_history(study)
                  progress_over_time.write_html(self.folder_saves+"/study_"+self.name_study+"/progress_over_time.html")
                  param_influence = optuna.visualization.plot_slice(study)
                  param_influence.write_html(self.folder_saves+"/study_"+self.name_study+"/param_influence.html")
                  for key, value in study.best_trial.params.items():
                        optuna.visualization.plot_slice(study, [key]).write_html(self.folder_saves+"/study_"+self.name_study+"/"+key+"_influence.html")

def conduct_fine_tuning(objective, name_study, timeout, n_trials, save_study=None, n_jobs=2, folder_saves="Saves_grand_slamin", type_of_task="regression"):
      optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
      callback_item = SaveStudyOptuna(name_study, folder_saves)
      if save_study==None:
            if type_of_task == "regression":
                  study = optuna.create_study(direction="minimize")
            else:
                  study = optuna.create_study(direction="maximize")
            sampler = optuna.samplers.TPESampler()
            study.sampler = sampler
      else:
            study = save_study
            sampler = optuna.samplers.TPESampler()
            study.sampler = sampler
      study.optimize(objective, timeout=timeout, n_trials=n_trials, n_jobs=n_jobs, callbacks=[callback_item])
      pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
      complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
      print("Study statistics: ")
      print("  Number of finished trials: ", len(study.trials))
      print("  Number of pruned trials: ", len(pruned_trials))
      print("  Number of complete trials: ", len(complete_trials))

      print("Best trial for study_"+name_study+" :")
      trial = study.best_trial
      print("  Value: ", trial.value)

      best_params = trial.params
      print("  Params: ")
      for key, value in best_params.items():
            print("    {}: {}".format(key, value))

      with open(folder_saves+"/study_"+name_study+"/best_params.pkl", 'wb') as handle:
            pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_results(dict_list, best_model, d_results, name_study, dict_params, trial, folder_saves, ind_repeat):
        try:
              number_trial = trial.number
        except:
              number_trial = 0
        if not(os.path.exists(folder_saves)):
                os.mkdir(folder_saves)
        if not(os.path.exists(folder_saves+"/study_"+name_study)):
                os.mkdir(folder_saves+"/study_"+name_study)
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial))):
                os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial))
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat))):
                os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat))
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/history")):
                os.mkdir(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/history")
        if not(os.path.exists(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/params.json")):
                with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/params.json", "w") as outfile:
                    json.dump(dict_params, outfile)

        l_params_names = np.sort(list(dict_params.keys()))
        l_params_values = [dict_params[l_params_names[i]] for i in range(len(l_params_names))]

        fig = go.Figure()
        for key_list in dict_list:
            l_names = key_list.split("/")
            name_save = l_names[0]
            name_plot = l_names[1]
            fig.add_trace(go.Scatter(x=np.arange(len(dict_list[key_list])), y=dict_list[key_list], name=name_plot))
            with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/"+name_save+".npy", 'wb') as f:
                np.save(f, dict_list[key_list])
            if name_plot=="Learning rate":
                fig.add_trace(go.Scatter(x=np.arange(len(dict_list[key_list])), y=2/dict_list[key_list], name="2/lr"))

        fig.update_layout(title="Summary for "+name_study)
        fig.write_html(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/summary.html")

        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/results.json", "w") as outfile:
                json.dump(d_results, outfile)

        path_model = folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/model"
        torch.save(best_model.state_dict(), path_model)
        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/deleted_main.npy", 'wb') as f:
                np.save(f, best_model.deleted_main)
        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/deleted_times_main.npy", 'wb') as f:
                np.save(f, best_model.deleted_times_main)
        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/deleted_inter.npy", 'wb') as f:
                np.save(f, best_model.deleted_inter)
        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/deleted_times_inter.npy", 'wb') as f:
                np.save(f, best_model.deleted_times_inter)              

        with open(folder_saves+"/study_"+name_study+"/trial_"+str(number_trial)+"/repeat_"+str(ind_repeat)+"/test_save_done.npy", 'wb') as f:
                np.save(f, True)

def delete_models(name_study, folder_saves, n_repeats, type_of_task, delete_pass=False):
      d_repeats = {}
      list_dir = os.listdir(folder_saves+"/study_"+name_study)
      trial_folders = np.array([x for x in list_dir if "trial" in x])
      print(trial_folders)
      l_metric = []
      l_to_keep = []
      for i in range(len(trial_folders)):
            model_folder = trial_folders[i]
            try:
                  validation_metric = 0
                  for ind_repeat in range(n_repeats):
                        with open(folder_saves+"/study_"+name_study+"/"+model_folder+"/repeat_"+str(ind_repeat)+"/test_save_done.npy", 'rb') as f:
                              test_save_done = np.load(f)
                        with open(folder_saves+"/study_"+name_study+"/"+model_folder+"/repeat_"+str(ind_repeat)+"/results.json", "r") as f:
                              dict_results = json.load(f)
                        if type_of_task=="classification":
                              validation_metric += dict_results["val_acc"]/n_repeats
                        elif type_of_task == "regression":
                              validation_metric += dict_results["val_mse"]/n_repeats
                  l_metric.append(validation_metric)
                  l_to_keep.append(i)
            except:
                  if delete_pass:
                        shutil.rmtree(folder_saves+"/study_"+name_study+"/"+model_folder)
                  else:
                        print("pass")
                        pass
      trial_folders = trial_folders[l_to_keep]
      if len(l_metric)>0:
            if type_of_task == "regression":
                  best_metric_ind = np.argmin(l_metric)
            elif type_of_task == "classification":
                  best_metric_ind = np.argmax(l_metric)
            for i in range(len(trial_folders)):
                  if i!=best_metric_ind:
                        try:
                              shutil.rmtree(folder_saves+"/study_"+name_study+"/"+trial_folders[i])
                              print(trial_folders[i]+" deleted")
                        except:
                              pass

def read_results(name_study, ind_repeat, type_of_task = "classification", folder_saves = "Saves_grand_slamin"):
      try:
            name_trial = os.listdir(folder_saves+"/study_"+name_study)
      except:
            print("The study doesn't exist,", folder_saves+"/study_"+name_study)

      name_trial = [x for x in name_trial if "trial" in x]
      if len(name_trial)>1:
            print("More than one model found")
      if len(name_trial)==0:
            print("No model found")

      if "best_trial" in name_trial:
            name_trial = "best_trial"
      else:
            name_trial = name_trial[0]
      try:
            with open(folder_saves+"/study_"+name_study+"/"+name_trial+"/params.json", "r") as f:
                  dict_params = json.load(f)
      except:
            print("couldn't read dict_params")

      with open(folder_saves+"/study_"+name_study+"/"+name_trial+"/repeat_"+str(ind_repeat)+"/results.json", "r") as f:
            dict_results = json.load(f)

      if type_of_task == "classification":
            in_sample_metric = np.array([dict_results["train_acc"], dict_results["train_auc"]])
            validation_metric = np.array([dict_results["val_acc"], dict_results["val_auc"]])
            test_metric = np.array([dict_results["test_acc"], dict_results["test_auc"]])
      elif type_of_task == "regression":
            in_sample_metric = dict_results["train_mse"]
            validation_metric = dict_results["val_mse"]
            test_metric = dict_results["test_mse"]

      best_ep = dict_results["best_ep"]
      time_training = dict_results["time_training"]
      n_z_i = dict_results["n_z_i"]
      n_z_ij = dict_results["n_z_ij"]
      n_features_used = dict_results["n_features_used"]
      if "n_params" in dict_results:
            n_params = dict_results["n_params"]
      else:
            n_params = None
      return dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z_i, n_z_ij, n_features_used, n_params

# ----------------
# --- Datasets ---
# ----------------

def one_hot_encoding(num_classes):
      def function(tensor):
            return torch.nn.functional.one_hot(tensor, num_classes=num_classes)
      return function

def load_data_census(load_directory='./',
      filename='pdb2022tr.csv',
      remove_margin_of_error_variables=True,
      remove_census_2020_variables=True,
      ):
      """Loads Census data, and retrieves covariates and responses.
      Args:
            load_directory: Data directory for loading Census file, str.
            filename: file to load, default is 'pdb2019trv3_us.csv'.
            remove_margin_of_error_variables: whether to remove margin of error variables, bool scaler.
      Returns:
            df_X, covariates, pandas dataframe.
            df_y, target response, pandas dataframe.
      """
      file = os.path.join(load_directory, filename)
      df = pd.read_csv(file, encoding = "ISO-8859-1")
      df = df.set_index('GIDTR')
      # Drop location variables
      drop_location_variables = ['State', 'State_name', 'County', 'County_name', 'Tract', 'Flag', 'AIAN_LAND']
      df = df.drop(drop_location_variables, axis=1)
      target_response = 'Self_Response_Rate_ACS_16_20'
      # Remove extra response variables
      if remove_margin_of_error_variables:
            df = df[np.array([c for c in df.columns if 'MOE' not in c])]
      if remove_census_2020_variables:
            drop_census_2020_variables = [col for col in df.columns if "CEN_2020" in col]
            df = df.drop(drop_census_2020_variables, axis=1)
      # Change types of covariate columns with dollar signs in their values e.g. income, housing price
      df[df.select_dtypes('object').columns] = df[df.select_dtypes('object').columns].replace('[\$,]', '', regex=True).astype(np.float64)
      # Remove entries with missing predictions
      df = df.dropna(subset=[target_response])
      df_y = df[[target_response]]
      df_X = df.drop([target_response], axis=1)
      cols_X = [col.split("_ACS_16_20")[0] for col in df_X.columns]
      df_X.columns = cols_X
      pct_cols = [col for col in df_X.columns if "pct" in col]
      avg_cols = [col for col in df_X.columns if "avg" in col and "pct" not in col]
      additional_cols = [
            "Num_BGs_in_Tract",
            "LAND_AREA",
            "Tot_Population",
            "Median_Age",
            "Civ_labor_16plus",
            "Civ_labor_16_24",
            "Civ_labor_25_44",
            "Civ_labor_45_64",
            "Civ_Noninst_Pop",
            "Civ_noninst_pop_U19",
            "Civ_noninst_pop1964",
            "Civ_noninst_pop_65P",
            "Population_age_3_4",
            "Pop_in_HHD",
            "Med_HHD_Inc",
            "Tot_Housing_Units",
            "Med_House_Value",
      ]
      df_X = df_X[additional_cols+pct_cols+avg_cols]
      return df_X, df_y

def get_dataset(name_dataset, n_kept_train = -1, device = "cpu", type_scaler = "std", type_embedding="layer"):
      if name_dataset in ["bikesharing", "microsoft", "year", "census"]:
            if name_dataset in ["bikesharing"]:
                  dataset = pd.read_csv("../data_regression/Bike-Sharing-Dataset/hour.csv")
                  dataset = dataset.drop(["instant","dteday","casual","registered"], axis = 1)
                  dataset_numpy = dataset.to_numpy()
                  y_numpy = dataset_numpy[:,-1]
                  dataset_numpy = dataset_numpy[:,:-1]
                  p_features = dataset_numpy.shape[1]
            if name_dataset in ["census"]:
                  df_X, df_y = load_data_census()
                  subset = [
                        "pct_Prs_Blw_Pov_Lev",
                        "pct_College",
                        "pct_Not_HS_Grad",
                        "pct_Pop_5_17",
                        "pct_Pop_18_24",
                        "pct_Pop_25_44",
                        "pct_Pop_45_64",
                        "pct_Pop_65plus",
                        "pct_Hispanic",
                        "pct_NH_White_alone",
                        "pct_NH_Blk_alone",
                        "pct_ENG_VW",
                        "pct_Othr_Lang",
                        "pct_Diff_HU_1yr_Ago",
                        "pct_Sngl_Prns_HHD",
                        "pct_Female_No_SP",
                        "pct_Rel_Under_6",
                        "pct_Vacant_Units",
                        "pct_Renter_Occp_HU",
                        "pct_Owner_Occp_HU",
                        "pct_Single_Unit",
                        "pct_HHD_Moved_in",
                        "pct_NO_PH_SRVC",
                        "pct_HHD_No_Internet",
                        "pct_HHD_w_Broadband",
                        "pct_Pop_w_BroadComp",
                        "pct_MrdCple_HHD",
                        "pct_NonFamily_HHD",
                        "pct_MLT_U2_9_STRC",
                        "pct_MLT_U10p",
                        "pct_One_Health_Ins",
                        "pct_TwoPHealthIns",
                        "pct_No_Health_Ins",
                        "avg_Tot_Prns_in_HHD",
                        "avg_Agg_HH_INC",
                        "avg_Agg_House_Value",
                        "Tot_Population",
                        "Civ_labor_16plus",
                        "pct_Civ_emp_16p",
                        ]
                  df_X = df_X[subset]
                  dataset_numpy = df_X.to_numpy()
                  y_numpy = df_y.to_numpy()[:,0]
                  p_features = dataset_numpy.shape[1]
            if name_dataset in ["year"]:
                  dataset = pd.read_csv("../data_regression/YearPredictionMSD.txt", header=None)
                  dataset_numpy = dataset.to_numpy()
                  y_numpy = dataset_numpy[:,0]
                  dataset_numpy = dataset_numpy[:,1:]
                  p_features = dataset_numpy.shape[1]
            if name_dataset in ["microsoft"]:
                  # Read data
                  dataset_train = pd.read_csv("../data_regression/MSLR-WEB10K/Fold1/train.txt", header=None, delimiter=" ").drop(columns=[138])
                  dataset_val = pd.read_csv("../data_regression/MSLR-WEB10K/Fold1/vali.txt", header=None, delimiter=" ").drop(columns=[138])
                  dataset_test = pd.read_csv("../data_regression/MSLR-WEB10K/Fold1/test.txt", header=None, delimiter=" ").drop(columns=[138])
                  # Divide X and y
                  train_dataset_numpy = dataset_train.iloc[:,1:]
                  val_dataset_numpy = dataset_val.iloc[:,1:]
                  test_dataset_numpy = dataset_test.iloc[:,1:]
                  train_y_numpy = dataset_train.iloc[:,0].to_numpy()
                  val_y_numpy = dataset_val.iloc[:,0].to_numpy()
                  test_y_numpy = dataset_test.iloc[:,0].to_numpy()
                  # Process the data
                  train_dataset_numpy = train_dataset_numpy.applymap(lambda x: x.split(":", 1)[-1])
                  train_dataset_numpy = train_dataset_numpy.astype(float)
                  val_dataset_numpy = val_dataset_numpy.applymap(lambda x: x.split(":", 1)[-1])
                  val_dataset_numpy = val_dataset_numpy.astype(float)
                  test_dataset_numpy = test_dataset_numpy.applymap(lambda x: x.split(":", 1)[-1])
                  test_dataset_numpy = test_dataset_numpy.astype(float)
                  # Drop the first column (query id)
                  train_dataset_numpy = train_dataset_numpy.drop(columns=1)
                  val_dataset_numpy = val_dataset_numpy.drop(columns=1)
                  test_dataset_numpy = test_dataset_numpy.drop(columns=1)
                  # Convert to numpy
                  train_dataset_numpy = train_dataset_numpy.to_numpy()
                  val_dataset_numpy = val_dataset_numpy.to_numpy()
                  test_dataset_numpy = test_dataset_numpy.to_numpy()
                  p_features = train_dataset_numpy.shape[1]

            meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(p_features)}

            if n_kept_train!=-1:
                  if name_dataset in ["microsoft"]:
                        train_dataset_numpy = train_dataset_numpy[:n_kept_train]
                        train_y_numpy = train_y_numpy[:n_kept_train]
                        val_dataset_numpy = val_dataset_numpy[:n_kept_train]
                        val_y_numpy = val_y_numpy[:n_kept_train]
                        test_dataset_numpy = test_dataset_numpy[:n_kept_train]
                        test_y_numpy = test_y_numpy[:n_kept_train]
                  else:
                        dataset_numpy = dataset_numpy[:n_kept_train]
                        y_numpy = y_numpy[:n_kept_train]

            if not(name_dataset in ["microsoft"]):
                  if name_dataset in ["year"]:
                        train_val_dataset_numpy, test_dataset_numpy, train_val_y_numpy, test_y_numpy = dataset_numpy[:463715], dataset_numpy[463715:], y_numpy[:463715], y_numpy[463715:]
                  else:
                        train_val_dataset_numpy, test_dataset_numpy, train_val_y_numpy, test_y_numpy = train_test_split(dataset_numpy, y_numpy, test_size=0.20, random_state=0)
                  train_dataset_numpy, val_dataset_numpy, train_y_numpy, val_y_numpy = train_test_split(train_val_dataset_numpy, train_val_y_numpy, test_size=0.20/0.90, random_state=1)
            imp_mean = SimpleImputer()
            imp_mean.fit(train_dataset_numpy)
            train_dataset_numpy = imp_mean.transform(train_dataset_numpy)
            val_dataset_numpy = imp_mean.transform(val_dataset_numpy)
            test_dataset_numpy = imp_mean.transform(test_dataset_numpy)
            for i in range(p_features):
                  if type_scaler == "std":
                        scaler_x = StandardScaler()
                  if type_scaler == "max":
                        scaler_x = MinMaxScaler()
                  scaler_x.fit(train_dataset_numpy[:,[i]])
                  train_dataset_numpy[:,[i]] = scaler_x.transform(train_dataset_numpy[:,[i]])
                  val_dataset_numpy[:,[i]] = scaler_x.transform(val_dataset_numpy[:,[i]])
                  test_dataset_numpy[:,[i]] = scaler_x.transform(test_dataset_numpy[:,[i]])
                  meta_info["X" + str(i + 1)]['scaler'] = scaler_x

            if type_scaler == "std":
                  scaler_y = StandardScaler()
            elif type_scaler == "max":
                  scaler_y = MinMaxScaler()
            scaler_y.fit(train_y_numpy[:, np.newaxis])
            train_y_numpy = scaler_y.transform(train_y_numpy[:, np.newaxis])[:,0]
            val_y_numpy = scaler_y.transform(val_y_numpy[:, np.newaxis])[:,0]
            test_y_numpy = scaler_y.transform(test_y_numpy[:, np.newaxis])[:,0]

            meta_info["Y"] = {}
            meta_info["Y"]["type"] = "target"
            meta_info["Y"]['scaler'] = scaler_y

            dataset_train = torch.Tensor(train_dataset_numpy)
            dataset_val = torch.Tensor(val_dataset_numpy)
            dataset_test = torch.Tensor(test_dataset_numpy)
            corres_y_train = torch.Tensor(train_y_numpy)
            corres_y_val = torch.Tensor(val_y_numpy)
            corres_y_test = torch.Tensor(test_y_numpy)

            corres_y_train = corres_y_train.float()
            corres_y_val = corres_y_val.float()
            corres_y_test = corres_y_test.float()

            dataset_train = dataset_train[None, :]
            dataset_val = dataset_val[None, :]
            dataset_test = dataset_test[None, :]

            corres_y_train = corres_y_train[None, :]
            corres_y_val = corres_y_val[None, :]
            corres_y_test = corres_y_test[None, :]
      else:
            if name_dataset in ["gisette", "madelon"]:
                  data_train = np.loadtxt("../data_classification/"+name_dataset+"/"+name_dataset+"_train.data")
                  data_valid = np.loadtxt("../data_classification/"+name_dataset+"/"+name_dataset+"_valid.data")
                  y_train = np.loadtxt("../data_classification/"+name_dataset+"/"+name_dataset+"_train.labels")
                  y_valid = np.loadtxt("../data_classification/"+name_dataset+"/"+name_dataset+"_valid.labels")
                  dataset_numpy = np.concatenate([data_train,data_valid])
                  y_numpy = np.concatenate([y_train,y_valid])
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["dorothea"]:
                  dico_data = scipy.io.loadmat("../data_classification/dorothea/dorothea.mat")
                  data_train = dico_data["Xtrain"]
                  data_valid = dico_data["Xvalid"]
                  y_train = dico_data["ytrain"][:,0]
                  y_valid = dico_data["yvalid"][:,0]
                  dataset_numpy = np.concatenate([data_train,data_valid])
                  y_numpy = np.concatenate([y_train,y_valid])
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["mice_protein"]:
                  dataset = pd.read_csv("../data_classification/mice_protein/Data_Cortex_Nuclear.csv")
                  dataset = dataset.iloc[:,1:]
                  dataset_numpy = pd.get_dummies(dataset.iloc[:,:-1]).to_numpy()
                  y_numpy = dataset.iloc[:,-1].to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["activity"]:
                  dataset = pd.read_csv("../data_classification/activity/data.txt", header=None)
                  y_numpy = dataset.iloc[:,-1].to_numpy()
                  dataset_numpy = dataset.iloc[:,1:-1].to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["mfeat"]:
                  dataset = pd.DataFrame()
                  n_features = 0
                  for part_data_path in ["mfeat-fac", "mfeat-fou", "mfeat-kar", "mfeat-mor", "mfeat-pix", "mfeat-zer"]:
                        new_dataset = pd.read_csv("../data_classification/mfeat/"+part_data_path, header=None, delim_whitespace=True)
                        n_new_features = new_dataset.shape[1]
                        new_dataset = new_dataset.rename(columns={i:i+n_features for i in range(n_new_features)})
                        n_features += n_new_features
                        dataset = pd.concat([dataset, new_dataset], axis=1)
                  y_numpy = np.arange(10)
                  y_numpy = np.repeat(y_numpy, 200)
                  dataset_numpy = dataset.to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["online"]:
                  dataset = pd.read_csv("../data_classification/OnlineNewsPopularity/OnlineNewsPopularity.csv").iloc[:,2:]
                  y_numpy = dataset.iloc[:,-1].to_numpy()
                  dataset_numpy = dataset.iloc[:,:-1].to_numpy()
                  y_numpy[y_numpy<1400] = 0
                  y_numpy[y_numpy>=1400] = 1
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["covertype"]:
                  dataset = pd.read_csv("../data_classification/covtype.data", header=None)
                  y_numpy = dataset.iloc[:,-1].to_numpy()
                  dataset_numpy = dataset.iloc[:,:-1].to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["miniboone"]:
                  dataset = pd.read_csv("../data_classification/MiniBooNE.txt", delim_whitespace=True, skiprows=1, header=None)
                  first_row = pd.read_csv("../data_classification/MiniBooNE.txt", nrows=1, header=None, delim_whitespace=True)
                  n_signal = first_row.iloc[0,0]
                  n_background = first_row.iloc[0,1]
                  y_numpy = np.zeros(n_signal+n_background)
                  y_numpy[:n_signal] = 1
                  dataset_numpy = dataset.to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["qsar"]:
                  dataset = pd.read_csv("../data_classification/qsar_oral_toxicity.csv", header=None, sep=";")
                  y_numpy = dataset.iloc[:,-1].to_numpy()
                  dataset_numpy = dataset.iloc[:,:-1].to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["taiwanese"]:
                  dataset = pd.read_csv("../data_classification/taiwanese.csv")
                  y_numpy = dataset.iloc[:,0].to_numpy()
                  dataset_numpy = dataset.iloc[:,1:].to_numpy()
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["liberty"]:
                  dataset = pd.read_csv("../data_classification/liberty.csv", index_col=0)
                  dataset = pd.get_dummies(dataset)
                  y_numpy = dataset.iloc[:,0].to_numpy()
                  dataset_numpy = dataset.iloc[:,1:].to_numpy()
                  y_numpy[y_numpy!=0] = 1
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}
            elif name_dataset in ["adult"]:
                  l_categorical = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
                  dataset = pd.read_csv("../data_classification/"+name_dataset+".csv", index_col=0)
                  dataset["target"] = dataset["target"] - np.min(dataset["target"])
                  y_numpy = dataset["target"].to_numpy()
                  dataset = dataset.iloc[:,:-1]
                  l_columns = list(dataset.columns)
                  l_continuous = [x for x in l_columns if x not in l_categorical]
                  dataset_cat = dataset[l_categorical]
                  dataset = dataset[l_continuous]
                  dataset_numpy = dataset.to_numpy()
                  dataset_cat_numpy = dataset_cat.to_numpy()
                  dataset_cat_numpy = OrdinalEncoder().fit_transform(dataset_cat_numpy)
                  meta_info = {"X" + str(i + 1):{'type':'categorical'} for i in range(dataset_cat_numpy.shape[1])}
                  for i in range(dataset_numpy.shape[1]):
                        meta_info["X"+str(i + 1 + dataset_cat_numpy.shape[1])] = {'type':'continuous'}
                  dataset_numpy = np.hstack([dataset_cat_numpy, dataset_numpy])
            elif name_dataset in ["churn"]:
                  l_categorical = ["state", "voice mail plan", "area code", "international plan"]
                  dataset = pd.read_csv("../data_classification/"+name_dataset+".csv", index_col=0)
                  dataset = dataset.drop(["phone number"], axis = 1)
                  dataset["target"] = dataset["target"] - np.min(dataset["target"])
                  y_numpy = dataset["target"].to_numpy()
                  dataset = dataset.iloc[:,:-1]
                  l_columns = list(dataset.columns)
                  l_continuous = [x for x in l_columns if x not in l_categorical]
                  dataset_cat = dataset[l_categorical]
                  dataset = dataset[l_continuous]
                  dataset_numpy = dataset.to_numpy()
                  dataset_cat_numpy = dataset_cat.to_numpy()
                  dataset_cat_numpy = OrdinalEncoder().fit_transform(dataset_cat_numpy)
                  meta_info = {"X" + str(i + 1):{'type':'categorical'} for i in range(dataset_cat_numpy.shape[1])}
                  for i in range(dataset_numpy.shape[1]):
                        meta_info["X"+str(i + 1 + dataset_cat_numpy.shape[1])] = {'type':'continuous'}
                  dataset_numpy = np.hstack([dataset_cat_numpy, dataset_numpy])
            else:
                  dataset = pd.read_csv("../data_classification/"+name_dataset+".csv", index_col=0)
                  dataset["target"] = dataset["target"] - np.min(dataset["target"])
                  dataset_numpy = dataset.to_numpy()
                  y_numpy = dataset_numpy[:,-1]
                  dataset_numpy = dataset_numpy[:,:-1]
                  meta_info = {"X" + str(i + 1):{'type':'continuous'} for i in range(dataset_numpy.shape[1])}

            enc = LabelEncoder()
            enc.fit(y_numpy)
            y_numpy = enc.transform(y_numpy)

            # enc = OrdinalEncoder()
            # enc.fit(y_numpy[:,np.newaxis])
            # y_numpy = enc.transform(y_numpy[:,np.newaxis])[:,0]
            
            meta_info["Y"] = {}
            meta_info["Y"]["type"] = "target"
            meta_info["Y"]['values'] = enc.classes_.tolist() #enc.categories_[0].tolist()

            if n_kept_train!=-1:
                  dataset_numpy = dataset_numpy[:n_kept_train]
                  y_numpy = y_numpy[:n_kept_train]

            if name_dataset in ["gisette", "madelon", "dorothea"]:
                  l_ind_train_val = np.arange(y_train.shape[0])
                  l_ind_test = np.arange(y_valid.shape[0])+y_train.shape[0]
                  train_val_dataset_numpy = dataset_numpy[l_ind_train_val]
                  test_dataset_numpy = dataset_numpy[l_ind_test]
                  train_val_y_numpy = y_numpy[l_ind_train_val]
                  test_y_numpy = y_numpy[l_ind_test]
            else:
                  train_val_dataset_numpy, test_dataset_numpy, train_val_y_numpy, test_y_numpy = train_test_split(dataset_numpy, y_numpy, test_size=0.20, random_state=0, stratify=y_numpy)
            train_dataset_numpy, val_dataset_numpy, train_y_numpy, val_y_numpy = train_test_split(train_val_dataset_numpy, train_val_y_numpy, test_size=0.20/0.90, random_state=1, stratify=train_val_y_numpy)
            imp_mean = SimpleImputer()
            imp_mean.fit(train_dataset_numpy)
            train_dataset_numpy = imp_mean.transform(train_dataset_numpy)
            val_dataset_numpy = imp_mean.transform(val_dataset_numpy)
            test_dataset_numpy = imp_mean.transform(test_dataset_numpy)
            for i in range(dataset_numpy.shape[1]):
                  if type_scaler == "std":
                        scaler_x = StandardScaler()
                        scaler_y = None
                  elif type_scaler == "max":
                        scaler_x = MinMaxScaler()
                        scaler_y = None
                  if meta_info["X"+str(i + 1)]["type"]=="categorical":
                        scaler_x = None
                        scaler_y = None
                        meta_info["X" + str(i + 1)]['n_cat_in'] = len(np.unique(dataset_numpy[:,i]))
                        if type_embedding=="layer":
                              meta_info["X" + str(i + 1)]['n_cat_out'] = 2 #(1+meta_info["X" + str(i + 1)]['n_cat_in'])//4 #meta_info["X" + str(i + 1)]['n_cat_in']
                              meta_info["X" + str(i + 1)]['encoder'] = torch.nn.Embedding(meta_info["X" + str(i + 1)]['n_cat_in'], meta_info["X" + str(i + 1)]['n_cat_out'], max_norm = 1.0)
                        elif type_embedding=="one_hot":
                              meta_info["X" + str(i + 1)]['n_cat_out'] = meta_info["X" + str(i + 1)]['n_cat_in']
                              meta_info["X" + str(i + 1)]['encoder'] = one_hot_encoding(num_classes=meta_info["X" + str(i + 1)]['n_cat_in'])
                        else:
                              print("Need to specify type_embedding!!")
                  else:
                        scaler_x.fit(train_dataset_numpy[:,[i]])
                        train_dataset_numpy[:,[i]] = scaler_x.transform(train_dataset_numpy[:,[i]])
                        val_dataset_numpy[:,[i]] = scaler_x.transform(val_dataset_numpy[:,[i]])
                        test_dataset_numpy[:,[i]] = scaler_x.transform(test_dataset_numpy[:,[i]])
                  meta_info["X" + str(i + 1)]['scaler'] = scaler_x
            dataset_train = torch.Tensor(train_dataset_numpy)
            dataset_val = torch.Tensor(val_dataset_numpy)
            dataset_test = torch.Tensor(test_dataset_numpy)
            corres_y_train = torch.Tensor(train_y_numpy)
            corres_y_val = torch.Tensor(val_y_numpy)
            corres_y_test = torch.Tensor(test_y_numpy)
            if len(np.unique(y_numpy))>2:
                  corres_y_train = corres_y_train.long()
                  corres_y_val = corres_y_val.long()
                  corres_y_test = corres_y_test.long()
            else:
                  corres_y_train = corres_y_train.float()
                  corres_y_val = corres_y_val.float()
                  corres_y_test = corres_y_test.float()

            dataset_train = dataset_train[None, :]
            dataset_val = dataset_val[None, :]
            dataset_test = dataset_test[None, :]

            corres_y_train = corres_y_train[None, :]
            corres_y_val = corres_y_val[None, :]
            corres_y_test = corres_y_test[None, :]

      dataset_train = dataset_train.to(device)
      dataset_val = dataset_val.to(device)
      dataset_test = dataset_test.to(device)
      corres_y_train = corres_y_train.to(device)
      corres_y_val = corres_y_val.to(device)
      corres_y_test = corres_y_test.to(device)

      return dataset_train, dataset_val, dataset_test, corres_y_train, corres_y_val, corres_y_test, meta_info

# -------------------
# --- Experiments ---
# -------------------

def convert_to_small_str(number):
      if int(number)==number:
            return str(int(number))
      if len(str(number))>5:
            power_of_ten = -int(np.floor(np.log(abs(number))/np.log(10)))
            number = number*10**(power_of_ten)
            number = np.round(number, 3)
            return str(number)+"e-"+str(power_of_ten)
      return str(number)

def get_name_study(lr=-1, name_dataset="spambase", depth="7", n_trees=3, n_epochs=10000, temperature=1.0, batch_size_SGD=64, n_train_kept=500, optimizer_name="SGD", type_decay=None, gamma_lr_decay=None, T_max_cos=None, eta_min_cos=None, start_lr_decay=None, end_lr_decay=None, test_early_stopping=0, test_compute_accurate_in_sample_loss = 0, n_repeats = 3, warmup_steps=100, n_features_kept=-1, type_scaler="std", patience=50, gamma=1.0, selection_reg=0.1, entropy_reg=0.01, hierarchy="none", l2_reg = 0, metric_early_stopping="val_loss", max_interaction_number=-1, period_milestones=25, alpha=1.0, test_different_lr=0, sel_penalization_in_hierarchy="none", lr_z=-1, type_embedding=""):
    name_study = "soft_add_depth_"+convert_to_small_str(depth)+"_T_"+convert_to_small_str(temperature)+"_trees_"+convert_to_small_str(n_trees)
    if optimizer_name!="Adam":
          name_study+="_"+optimizer_name
    if lr != -1:
          name_study += "_lr_"+convert_to_small_str(lr)
    if type_embedding!="":
      name_study += "embed_"+type_embedding
    name_study+="_"+convert_to_small_str(batch_size_SGD)
    name_study+="_"+name_dataset
    name_study+="_"+type_scaler
    if n_train_kept!=-1:
        name_study+="_"+convert_to_small_str(n_train_kept)
    if (n_features_kept!=-1):
        name_study+="_features_kept_"+convert_to_small_str(n_features_kept)
    if n_epochs!=1000:
        name_study+="_n_epochs_"+convert_to_small_str(n_epochs)
    if (type_decay!="None"):
        name_study+="_"+type_decay
        if type_decay=="linear":
                name_study+="_linear_start_"+convert_to_small_str(start_lr_decay)+"_end_"+convert_to_small_str(end_lr_decay)
        elif type_decay=="exponential":
                name_study+="_"+convert_to_small_str(gamma_lr_decay)
        elif type_decay=="cosine":
                name_study+="_T_max_"+convert_to_small_str(T_max_cos)+"_min_lr_"+convert_to_small_str(eta_min_cos)
        elif type_decay=="multi_lr":
                name_study+="_p_"+convert_to_small_str(period_milestones)+"_g_"+convert_to_small_str(gamma_lr_decay)
        elif type_decay=="ramp":
                name_study+="_"+convert_to_small_str(warmup_steps)
    if test_early_stopping:
        name_study+="_es"
        if patience!=-1:
                name_study+="_patience_"+convert_to_small_str(patience)
        name_study += "_"+metric_early_stopping
        if test_compute_accurate_in_sample_loss:
                name_study+="_accurate_loss"
    
    name_study+="_gamma_"+convert_to_small_str(gamma)
    name_study+="_sel_reg_"+convert_to_small_str(selection_reg)
    name_study+="_ent_reg_"+convert_to_small_str(entropy_reg)
    name_study+="_l2_reg_"+convert_to_small_str(l2_reg)
    if alpha!=1.0:
      name_study+="_alp_"+convert_to_small_str(alpha)
    name_study+="_hier_"+hierarchy
    name_study+="_inter"+convert_to_small_str(max_interaction_number)
    if test_different_lr!=0:
        name_study+="_diff_lr"
        if lr_z!=-1:
            name_study+="_"+convert_to_small_str(lr_z)
    if sel_penalization_in_hierarchy and hierarchy in ["strong", "weak"]:
        name_study+="_sel_reg_hier"
    if n_repeats!=1:
        name_study+="_"+convert_to_small_str(n_repeats)
    return name_study

print("Import utils done")
# %%
