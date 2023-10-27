#%%
from utils import *
from utils_grand_slamin import *
import plotly.graph_objs as go

#%%

folder_visualizations = "visualize_main_effects"

path_hierarchy_none = "Saves_grand_slamin_bikesharing_repeats_temp/study_soft_add_depth_4_T_1_trees_1_lr_0.01_64_bikesharing_std_multi_lr_p_25_g_0.9_es_patience_50_val_loss_gamma_1_sel_reg_1.0e-4_ent_reg_0.1_l2_reg_1e-05_hier_0_main_-1_inter-1_diff_lr_10"
path_hierarchy_strong = "Saves_grand_slamin_bikesharing_repeats_temp/study_soft_add_depth_4_T_1_trees_1_lr_0.01_64_bikesharing_std_multi_lr_p_25_g_0.9_es_patience_50_val_loss_gamma_1_sel_reg_1.0e-4_ent_reg_0.1_l2_reg_1e-05_hier_1_main_-1_inter-1_diff_lr_10"
path_hierarchy_weak = "Saves_grand_slamin_bikesharing_repeats_temp/study_soft_add_depth_4_T_1_trees_1_lr_0.01_64_bikesharing_std_multi_lr_p_25_g_0.9_es_patience_50_val_loss_gamma_1_sel_reg_1.0e-4_ent_reg_0.1_l2_reg_1e-05_hier_2_main_-1_inter-1_diff_lr_10"

path_saves = [path_hierarchy_none, path_hierarchy_strong, path_hierarchy_weak]

readable_labels = {
     0  : "season",
     1  : "year",
     2  : "month",
     3  : "hour",
     4  : "holiday",
     5  : "day of week",
     6  : "workday",
     7  : "weather",
     8  : "temperature",
     9 : "feels_like_temp",
     10 : "humidity",
     11 : "wind speed",
}

is_categorical = {
     0  : 1,
     1  : 1,
     2  : 1,
     3  : 1,
     4  : 1,
     5  : 1,
     6  : 1,
     7  : 1,
     8  : 0,
     9 : 0,
     10 : 0,
     11 : 0,
}

#type_graph = "heatmap"
type_graph = "markers"

d_plots_main = {}
d_plots_inter = {}

if not(os.path.exists(folder_visualizations)):
    os.mkdir(folder_visualizations)

for hierarchy in ["none", "strong", "weak"]:
    print("Computating graph for hierarchy =", hierarchy)
    path_study = path_saves[hierarchy]
    device = "cpu"
    bins = 100
    n_repeats = 10
    ind_repeat_inter = 0

    with open(path_study+"/best_trial/params.json", "r") as f:
        dict_params = json.load(f)

    dataset_train, dataset_val, dataset_test, corres_y_train, corres_y_val, corres_y_test, meta_info = get_dataset(dict_params["name_dataset"], dict_params["n_train_kept"], device, dict_params["type_scaler"])
    n_main_original = dataset_train.shape[-1]
    l_combinations_original = np.array(list(itertools.combinations(np.arange(n_main_original), 2)))
    interaction_terms = [tuple(x)for x in l_combinations_original]
    max_interaction_number = dict_params["max_interaction_number"]
    n_samples = dataset_train.shape[1]
    steps_per_epoch = np.ceil(n_samples/dict_params["batch_size_SGD"])
    X = torch.concatenate([dataset_train[0], dataset_val[0], dataset_test[0]], dim=0).numpy()
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    X_main = {}
    for i in range(n_main_original):
        if is_categorical[i]:
            X_main[i] = torch.Tensor(np.unique(X[:,i]))
        else:
            X_main[i] = torch.Tensor(np.linspace(X_min[i],X_max[i], bins))

    fmain_final = {}
    finteraction_final = {}
    for i in range(n_main_original):
        fmain_final[i] = np.zeros((0, X_main[i].shape[0]))
    for term in interaction_terms:
        finteraction_final[term] = np.zeros((0, X_main[term[0]].shape[0], X_main[term[1]].shape[0]))

    for ind_repeat in range(n_repeats):
        model = read_model(path_study, ind_repeat, device, steps_per_epoch)
        active_interaction_terms = [tuple(model.l_combinations[i].numpy()) for i in range(len(model.l_combinations))]
        fmain, finteraction, X_main, X_interaction = purify_model_for_visualization(model, X_main, interaction_terms, active_interaction_terms)
        for i in range(n_main_original):
            fmain_final[i] = np.concatenate([fmain_final[i], fmain[i][np.newaxis]])
        for term in interaction_terms:
            try:
                finteraction_final[term] = np.concatenate([finteraction_final[term], finteraction[term][np.newaxis]])
            except:
                import ipdb; ipdb.set_trace()

    # Print main
    for i in range(n_main_original):
        x_scaled = meta_info[f"X{i+1}"]["scaler"].inverse_transform(X_main[i][:,np.newaxis])[:,0]
        mean_output = np.mean(fmain_final[i], 0)
        mad_ouput = np.mean(np.abs(fmain_final[i]-mean_output), 0)
        d_plots_main[(i,hierarchy)] = [x_scaled, mean_output, mad_ouput]

# %%

for i in range(n_main_original):
    min_val_y = np.inf
    max_val_y = -np.inf
    for hierarchy in ["none", "strong", "weak"]:
        x_scaled, mean_output, mad_ouput = d_plots_main[(i,hierarchy)]
        min_val_y = min(min_val_y, np.min(mean_output-mad_ouput))
        max_val_y = max(max_val_y, np.max(mean_output+mad_ouput))
    range_length = max_val_y - min_val_y
    max_val_y += 0.1*range_length
    min_val_y -= 0.1*range_length
    print(readable_labels[i], min_val_y, max_val_y)
    for hierarchy in ["none", "strong", "weak"]:
        fig = go.Figure(layout_yaxis_range = [min_val_y, max_val_y])
        x_scaled, mean_output, mad_ouput = d_plots_main[(i,hierarchy)]

        fig.add_trace(go.Scatter(
                #name='Measurement',
                x=x_scaled,
                y=mean_output,
                mode='lines',
                #line=dict(color='rgb(31, 119, 180)'),
                showlegend=False,
                textfont_size=50
            ))
        fig.add_trace(go.Scatter(
                # name='Upper Bound',
                x=x_scaled,
                y=mean_output+mad_ouput,
                mode='lines',
                #marker=dict(color="#444"),
                marker = dict(color='rgb(55, 126, 247)'),
                line=dict(width=0),
                showlegend=False,
                textfont_size=50
            ))
        fig.add_trace(go.Scatter(
                # name='Lower Bound',
                x=x_scaled,
                y=mean_output-mad_ouput,
                #marker=dict(color="#444"),
                marker = dict(color='rgb(55, 126, 247)'),
                line=dict(width=0),
                mode='lines',
                #fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False,
                textfont_size=50
            ))
        fig.update_layout(
            xaxis_title=readable_labels[i],
            #title=f"Feature: {readable_labels[i]} for hierarchy = {hierarchy}",
            hovermode="x",
            font=dict(size=25)
        )
        fig.update_yaxes(title="Main effect")

        fig.write_html(f"{folder_visualizations}/feature_{readable_labels[i]}_hierarchy_{hierarchy}.html")
        fig.write_image(f"{folder_visualizations}/feature_{readable_labels[i]}_hierarchy_{hierarchy}.png", scale=5)

# %%
