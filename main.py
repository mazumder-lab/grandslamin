#%% Imports
import argparse
import time
import signal

signal.signal(signal.SIGVTALRM, lambda signum, frame: print("\n--- Time is over ---"))
#%%
start_time = time.time()

parser = argparse.ArgumentParser(description='Grand Slamin training')

# SHARED HYPERPARAMETERS
parser.add_argument('--n_epochs', type=int, default = 1000,
                    help='number of epochs for the training function')
parser.add_argument('--timeout', type=int, default = 72*60*60,
                    help='timeout for the fine tuning')
parser.add_argument('--n_trials', type=int, default = 1,
                    help='number of trials for the fine tuning')
parser.add_argument('--n_repeats', type=int, default = 1,
                    help='number of times we repeat the experiment for a given set of hyperparameters')
parser.add_argument('--name_dataset', type=str, default = "churn",
                    help='name of the dataset')
parser.add_argument('--n_train_kept', type=int, default = -1,
                    help='maximum number of samples kept in the training set')
parser.add_argument('--lr', type=float, default = -1,
                    help='if a value is given, no fine tuning is performed for the learning rate')
parser.add_argument('--min_lr', type=float, default = 1e-5,
                    help='lower bound of the gridsearch given to Optuna for the learning rate. If lr != -1, then this parameter is ignored')
parser.add_argument('--max_lr', type=float, default = 1e-1,
                    help='upper bound of the gridsearch given to Optuna for the learning rate. If lr != -1, then this parameter is ignored')
parser.add_argument('--folder_saves', type=str, default = "Saves_grand_slamin",
                    help='name of the folder where all the studies are being saved')
parser.add_argument('--type_scaler', type=str, default = "std",
                    help='type of scaling method (std or max)')
parser.add_argument('--type_of_task', type=str, default = "classification",
                    help='either classification or regression. In case of classification, n_trees is set to the number of possible classes')
parser.add_argument('--device', type=str, default = "None",
                    help='if None, then the best possible device will be used')
parser.add_argument('--metric_best_model', type=str, default = "auc",
                    help='either auc or acc')
parser.add_argument('--seed', type=int, default = 0,
                    help='seed for the training')
parser.add_argument('--batch_size_SGD', type=int, default = 64,
                    help='batch size for SGD')
parser.add_argument('--hierarchy', type=str, default = "none",
                    help='"weak" for weak hierarchy, "strong" for strong hierarchy, "none" for no hierarchy')
parser.add_argument('--depth', type=int, default = 4,
                    help='depth for grand_slamin')
parser.add_argument('--lr_z', type=float, default = -1,
                    help='if a value is given and test_diff_lr = 1, this learning rate is used for the z_i, otherwise lr/n_steps_per_epoch is used when test_diff_lr = 1')
parser.add_argument('--n_trees', type=int, default = 1,
                    help='number of trees (if greater than 1, we average the output of the ensemble of trees)')
parser.add_argument('--temperature', type=float, default = 1.0,
                    help='temperature for the tempered sigmoid (a value of 1 corresponds to the usual sigmoid function, the higher the temperature, the shaper the slope)')
parser.add_argument('--test_early_stopping', type=int, default = 1,
                    help='If test_early_stopping=1, the best model out of the n_epochs iterations is kept based on the validation loss. If test_early_stopping=1, the training loss is used')
parser.add_argument('--type_decay', type=str, default = "None",
                    help='criteria for the decay. Either "None" or "multi_lr". If type_decay = "None", then no decay is applied. If type_decay = "multi_lr", MultiStepLR decay is applied')
parser.add_argument('--gamma_lr_decay', type=float, default = 0.9,
                    help='learning rate decay for type_decay = "multi_lr"')
parser.add_argument('--warmup_steps', type=int, default = 100,
                    help='epoch where the learning rate reaches its maximum for type_decay = "ramp"')
parser.add_argument('--path_load_weights', type=str, default = "",
                    help='if a path is provided, it is used to initialize the weights of the model')
parser.add_argument('--test_compute_accurate_in_sample_loss', type=int, default = 0,
                    help='recomputes the in-sample loss exactly')
parser.add_argument('--n_features_kept', type=int, default = -1,
                    help='number of features kept')
parser.add_argument('--max_interaction_number', type=int, default = -1,
                    help='number of interaction features kept, -1 means all the features are kept')
parser.add_argument('--patience', type=int, default = 50,
                    help='patience for early stopping')
parser.add_argument('--gamma', type=float, default = 1.0,
                    help='gamma for SmoothStep')
parser.add_argument('--entropy_reg', type=float, default = 0.1,
                    help='regularizer factor for entropy penalization in Grand Slamin')
parser.add_argument('--selection_reg', type=float, default = 0.001,
                    help='regularizer factor for selection penalization in Grand Slamin')
parser.add_argument('--alpha', type=float, default = 1.0,
                    help='the selection loss is selection_loss_main + alpha * selection_loss_inter: if alpha > 1, we penalize the selection of interaction effects more')
parser.add_argument('--metric_early_stopping', type=str, default = "val_loss",
                    help='either val_loss or val_accuracy')
parser.add_argument('--l2_reg', type=float, default = 0.001,
                    help='regularizer factor for l2 penalization in Grand Slamin')
parser.add_argument('--period_milestones', type=int, default = 25,
                    help='period of the milestones for multi_lr scheduler')
parser.add_argument('--test_different_lr', type=int, default = 1,
                    help='if set to 1, then lr/steps_per_epoch is used for the weights corresponding to the z_i and z_ij. Otherwise, the regular learning rate is used')
parser.add_argument('--dense_to_sparse', type=int, default = 1,
                    help='if set to 1, then the weights of the model are eliminated during the training. Currently, only works with Adam, has to be set to 0 for another optimizer.')
parser.add_argument('--type_embedding', type=str, default = "one_hot",
                    help='type of embedding for the caterogical variables (one_hot or layer)')
parser.add_argument('--optimizer_name', type=str, default = "Adam",
                    help='Adam, SGD or Adagrad')

#%%
if __name__ == '__main__':
    arguments = parser.parse_args()
    print('Parsed arguments:', arguments)
    n_epochs = arguments.n_epochs
    timeout = arguments.timeout
    n_trials = arguments.n_trials
    name_dataset = arguments.name_dataset
    depth = arguments.depth
    n_trees = arguments.n_trees
    n_train_kept = arguments.n_train_kept
    learning_rate = arguments.lr
    temperature = arguments.temperature
    batch_size_SGD = arguments.batch_size_SGD
    test_early_stopping = arguments.test_early_stopping
    optimizer_name = arguments.optimizer_name
    min_lr = arguments.min_lr
    max_lr = arguments.max_lr
    type_decay = arguments.type_decay
    gamma_lr_decay = arguments.gamma_lr_decay
    path_load_weights = arguments.path_load_weights
    type_of_task = arguments.type_of_task
    test_compute_accurate_in_sample_loss = arguments.test_compute_accurate_in_sample_loss
    n_repeats = arguments.n_repeats
    folder_saves = arguments.folder_saves
    warmup_steps = arguments.warmup_steps
    n_features_kept = arguments.n_features_kept
    max_interaction_number = arguments.max_interaction_number
    type_scaler = arguments.type_scaler
    patience = arguments.patience
    hierarchy = arguments.hierarchy
    gamma = arguments.gamma
    entropy_reg = arguments.entropy_reg
    selection_reg = arguments.selection_reg
    alpha = arguments.alpha
    l2_reg = arguments.l2_reg
    metric_early_stopping = arguments.metric_early_stopping
    device = arguments.device
    period_milestones = arguments.period_milestones
    metric_best_model = arguments.metric_best_model
    test_different_lr = arguments.test_different_lr
    dense_to_sparse = arguments.dense_to_sparse
    seed = arguments.seed
    lr_z = arguments.lr_z
    type_embedding = arguments.type_embedding

    if name_dataset in ["churn", "adult"] and dense_to_sparse == 1:
        dense_to_sparse = 0
        print(" --- ")
        print("Dense to sparse training not implemented yet for datasets with categorical embedding layers, switching to dense training", flush=True)
        print(" --- ")

    # Necessary imports:
    from utils_grand_slamin import *
    from utils import *
    if device == "None":
        test_found_gpu = False
        try:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                test_found_gpu = True
        except:
            pass

        if torch.cuda.is_available():
            device = 'cuda'
            test_found_gpu = True
        if not(test_found_gpu):
            device = "cpu"

    print("--- Device =", device, "---")

    bias = True
    activation_func = tempered_sigmoid(temperature)

    dataset_train, dataset_val, dataset_test, corres_y_train, corres_y_val, corres_y_test, meta_info = get_dataset(name_dataset, n_train_kept, device, type_scaler, type_embedding)
    dataset_temp = (dataset_train, corres_y_train), (dataset_val, corres_y_val), (dataset_test, corres_y_test), meta_info
    n_main = dataset_train.shape[-1]
    if max_interaction_number != -1:
        n_interactions = max_interaction_number
    else:
        n_interactions = int(n_main*(n_main-1)/2)
    n_samples = dataset_train.shape[1]
        
    if type_of_task == "classification":
        if device!="cpu":
            p_output = len(np.unique(np.concatenate([corres_y_train[0].cpu(), corres_y_val[0].cpu(), corres_y_test[0].cpu()])))
        else:
            p_output = len(np.unique(np.concatenate([corres_y_train[0], corres_y_val[0], corres_y_test[0]])))
    else:
        p_output = 1
    
    name_study = get_name_study(lr=learning_rate, 
                                name_dataset=name_dataset, 
                                depth=depth, 
                                n_trees=n_trees, 
                                n_epochs=n_epochs,
                                temperature=temperature,
                                batch_size_SGD=batch_size_SGD,
                                n_train_kept=n_train_kept,
                                optimizer_name=optimizer_name, 
                                type_decay=type_decay,
                                gamma_lr_decay=gamma_lr_decay,
                                test_early_stopping=test_early_stopping,
                                test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss,
                                n_repeats=n_repeats,
                                warmup_steps=warmup_steps,
                                n_features_kept=n_features_kept,
                                type_scaler=type_scaler,
                                patience=patience,
                                gamma=gamma,
                                selection_reg=selection_reg,
                                entropy_reg=entropy_reg,
                                hierarchy=hierarchy,
                                l2_reg=l2_reg,
                                metric_early_stopping=metric_early_stopping,
                                max_interaction_number=max_interaction_number,
                                period_milestones=period_milestones,
                                alpha=alpha,
                                test_different_lr=test_different_lr,
                                lr_z=lr_z,
                                type_embedding=type_embedding)

    if batch_size_SGD == -1:
        batch_size_SGD = n_samples

    steps_per_epoch = np.ceil(n_samples/batch_size_SGD)
    test_deja_train = False
    n_trials_done = 0
    save_study_done = None
    if folder_saves in os.listdir():
        if ("study_"+name_study) in os.listdir(folder_saves):
            if len(os.listdir(folder_saves+"/study_"+name_study))>=1:
                try:
                    with open(folder_saves+"/study_"+name_study+"/save_study.pkl", "rb") as f:
                        save_study_done = pickle.load(f)
                    n_trials_done = len(save_study_done.trials)
                except:
                    pass

    if n_trials_done>=n_trials:
        test_deja_train = True
        try:
            list_dir = os.listdir(folder_saves+"/study_"+name_study)
            trial_folders = np.array([x for x in list_dir if "trial" in x])
            path_model = folder_saves+"/study_"+name_study+"/"+trial_folders[0]
            new_path_model =  folder_saves+"/study_"+name_study+"/best_trial"
            os.rename(path_model, new_path_model)
        except:
            pass
    else:
        n_trials = n_trials - n_trials_done
        if n_trials_done>0:
            print("Continuing existing study...")
        else:
            print("New study...")
        print(str(n_trials)+" trials restants")
    if test_deja_train:
        print("Training already done for "+name_study)
    else:
        if path_load_weights!="":
            list_models_loaded = True
            try:
                list_models_state_dict = torch.load(path_load_weights+"/best_trial/repeat_0/history/all_models.pth", map_location=device)
            except:
                params_model_0 = torch.load(path_load_weights, map_location=device)
                list_models_loaded = False
        def objective(trial):
            try:
                global dataset
            except:
                pass
            try:
                global meta_info
            except:
                pass
            if learning_rate!=-1:
                lr = learning_rate
            else:
                lr = trial.suggest_float("lr", min_lr, max_lr, log = True)
            print('Parsed arguments:', arguments)
            print("lr:", lr)
            if type_of_task== "classification":
                best_val_metric_best = -np.inf
            elif type_of_task=="regression":
                best_val_metric_best = np.inf
            best_val_metric_avg = 0
            for ind_repeat in range(n_repeats):
                signal.alarm(0)
                signal.alarm(24*60*60)
                print("Repeat", ind_repeat+1, "out of", n_repeats, flush=True)
                if type_of_task == "classification":
                        if p_output==2:
                            criterion = torch.nn.BCELoss().to(device)
                        else:
                            loss_fun = torch.nn.NLLLoss().to(device)
                            def criterion(x,y):
                                return loss_fun(torch.log(x+1e-6), y)
                elif type_of_task == "regression":
                    criterion = torch.nn.MSELoss().to(device)
                # Model initialization
                model = Grand_slamin(n_trees=n_trees, 
                    n_main=n_main, 
                    n_interactions=n_interactions, 
                    depth=depth, 
                    bias=bias,
                    activation_func=activation_func, 
                    p_output=p_output,
                    n_features_kept = n_features_kept,
                    weight_initializer = "glorot",
                    bias_initializer = "zeros",
                    hierarchy = hierarchy,
                    gamma = gamma,
                    alpha=alpha,
                    selection_reg = selection_reg,
                    entropy_reg=entropy_reg,
                    l2_reg=l2_reg,
                    test_different_lr=test_different_lr,
                    steps_per_epoch=steps_per_epoch,
                    dense_to_sparse=dense_to_sparse,
                    type_of_task=type_of_task,
                    lr_z=lr_z,
                    seed = seed+ind_repeat,
                    device = device,
                    meta_info=meta_info)
                dataset_train_main, dataset_val_main, dataset_test_main, dataset_train_inter, dataset_val_inter, dataset_test_inter, corres_y_train, corres_y_val, corres_y_test, meta_info = model.perform_screening(dataset_temp, max_interaction_number)
                dataset = (dataset_train_main, dataset_val_main, dataset_test_main, dataset_train_inter, dataset_val_inter, dataset_test_inter, corres_y_train, corres_y_val, corres_y_test, meta_info)
                if path_load_weights!="":
                    if list_models_loaded:
                        model.load_state_dict(list_models_state_dict[0])
                    else:
                        try:
                            model.load_state_dict(params_model_0)
                        except:
                            params_model_0["mask_features"] = model.mask_features
                            model.load_state_dict(params_model_0)
                    print("The weights have been loaded")
                model.to(device)
                optimizer = initialize_optimizer(test_different_lr, model, optimizer_name, steps_per_epoch, lr, lr_z)
                # Training
                time_before_training = time.time()
                d_results, best_val_metric, best_model, dict_list = train_grand_slamin(name_study=name_study, model=model, dataset=dataset, optimizer=optimizer, criterion=criterion, n_epochs=n_epochs, batch_size_SGD=batch_size_SGD, path_save=None, test_early_stopping=test_early_stopping, trial=trial, type_decay=type_decay, gamma_lr_decay=gamma_lr_decay, warmup_steps=warmup_steps, type_of_task=type_of_task, test_compute_accurate_in_sample_loss=test_compute_accurate_in_sample_loss, folder_saves=folder_saves, ind_repeat=ind_repeat, patience = patience, metric_early_stopping=metric_early_stopping, period_milestones=period_milestones)
                if type_of_task == "classification":
                    if metric_best_model == "acc":
                        best_val_metric = best_val_metric[0]
                    elif metric_best_model == "auc":
                        best_val_metric = best_val_metric[1]
                    else:
                        print("Error: wrong metric to pick the best model, either acc or auc")
                time_training = time.time()-time_before_training
                d_results["time_training"] = time_training
                best_val_metric_avg += best_val_metric/n_repeats
                if ((best_val_metric>=best_val_metric_best) and (type_of_task=="classification")) or ((best_val_metric<=best_val_metric_best) and (type_of_task=="regression")):
                    best_val_metric_best = best_val_metric
                best_model.to(device)

                #Saving of the results
                dict_params = vars(arguments)
                dict_params["lr"] = lr
                
                save_results(dict_list=dict_list, best_model=best_model, d_results=d_results, name_study=name_study, dict_params=dict_params, trial=trial, folder_saves=folder_saves, ind_repeat=ind_repeat)
            delete_models(name_study, folder_saves, n_repeats, type_of_task)
            return best_val_metric_avg

        if save_study_done!=None:
            conduct_fine_tuning(objective, name_study, timeout, n_trials, save_study_done, folder_saves=folder_saves, type_of_task=type_of_task)
        else:
            conduct_fine_tuning(objective, name_study, timeout, n_trials, folder_saves=folder_saves, type_of_task=type_of_task)
    
    delete_models(name_study, folder_saves, n_repeats, type_of_task, delete_pass=True)

    try:
        list_dir = os.listdir(folder_saves+"/study_"+name_study)
        trial_folders = np.array([x for x in list_dir if "trial" in x])
        path_model = folder_saves+"/study_"+name_study+"/"+trial_folders[0]
        new_path_model =  folder_saves+"/study_"+name_study+"/best_trial"
        os.rename(path_model, new_path_model)
    except:
        pass

    in_sample_metric_avg = 0
    validation_metric_avg = 0
    test_metric_avg = 0

    for ind_repeat in range(n_repeats):
        dict_params, in_sample_metric, validation_metric, test_metric, best_ep, time_training, n_z_i, n_z_ij, n_features_used, n_params = read_results(name_study, ind_repeat, type_of_task =type_of_task, folder_saves=folder_saves)
        in_sample_metric_avg += in_sample_metric/n_repeats
        validation_metric_avg += validation_metric/n_repeats
        test_metric_avg += test_metric/n_repeats

    print("Best params =", dict_params)
    print("Best epoch = "+ str(best_ep)+"/"+str(n_epochs-1))
    print("  ")
    if type_of_task=="regression":
        print("In-sample mse =", in_sample_metric_avg)
        print("Validation mse =", validation_metric_avg)
        print("Out-of-sample mse =", test_metric_avg)
    elif type_of_task=="classification":
        print("In-sample accuracy =", 100*in_sample_metric_avg[0],"%")
        print("Validation accuracy =", 100*validation_metric_avg[0],"%")
        print("Out-of-sample accuracy =", 100*test_metric_avg[0], "%")
        print("In-sample auc =", in_sample_metric_avg[1])
        print("Validation auc =", validation_metric_avg[1])
        print("Out-of-sample auc =", test_metric_avg[1])
    print("Training time = ", time_training)
    print("TOTAL TIME = ", time.time()-start_time)
# %%
