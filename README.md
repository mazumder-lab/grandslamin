# GRAND-SLAMIN’ Interpretable Additive Modeling with Structural Constraints

This is the offical repo of the Neurips 2023 paper **GRAND-SLAMIN’ Interpretable Additive Modeling with Structural Constraints**

## Requirements
This code has been tested with Python 3.10.11 and the following packages:
```
pandas==1.5.3
numpy==1.23.5
torch==2.0.0
torchaudio==2.0.0
torchvision==0.15.0
optuna==3.1.1
matplotlib==3.7.1
plotly==5.14.1
scikit-learn==1.2.2
scipy==1.10.0
tqdm==4.65.0
```

## Structure of the repo
Scripts for the experiments (name, finetuning, etc) are located in `utils.py`. Scripts for the GRAND-SLAMIN implementation are located in `utils_grand_slamin.py`.
To start traning a GRAND-SLAMIN model, you can run `main.py` specifying the hyperparameters you want as arguments. For example:
- `python main.py --name_dataset online --lr 0.005 --n_epochs 1000 --hierarchy strong --folder_saves Saves_grand_slamin --entropy_reg 0.01 --selection_reg 0.0002154434690031 --alpha 10.0 --l2_reg 0.0`
- `python main.py --name_dataset optdigits --hierarchy weak --lr 0.01 --n_epochs 1000 --folder_saves Saves_grand_slamin --entropy_reg 0.1 --selection_reg 4.641588833612772e-05 --l2_reg 1e-05`
- `python main.py --name_dataset mfeat --max_interaction_number 20000  --n_epochs 2000 --patience 500 --hierarchy none --metric_early_stopping val_accuracy --folder_saves Saves_grand_slamin --lr 5e-5 --gamma 0.1 --entropy_reg 0.1 --selection_reg 0.0 --lr_z 0.0005`

To visualize the main effects of a GRAND-SLAMIN model, you can run `visualize_main_effects.py` (you can specify the path of the model(s) you want to visualize in the script).

## Citing GRAND-SLAMIN'
If you find GRAND-SLAMIN useful in your research, please consider citing the following paper.

```
@inproceedings{
grandslamin,
title={{GRAND}-{SLAMIN}{\textquoteright} Interpretable Additive Modeling with Structural Constraints},
author={Ibrahim, Shibal and Afriat, Gabriel I. and Behdin, Kayhan and Mazumder, Rahul},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=F5DYsAc7Rt}
}
```