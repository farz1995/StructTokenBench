import os
import sys
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

# local imports
exc_dir = os.path.dirname(os.path.dirname(__file__)) # "src/"
sys.path.append(exc_dir)
exc_dir_baseline = os.path.join(os.path.abspath(exc_dir), "baselines")
all_baseline_names = glob(exc_dir_baseline + "/*")
for name in all_baseline_names:
    if name != "cheap_proteins":
        sys.path.append(os.path.join(exc_dir_baseline, name))
sys.path.append(exc_dir_baseline)

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager


def visualize_simscore_violin(sim_list, data_name_list, datasource):
    
    palette = sns.color_palette("husl", len(sim_list))
    palette = [mcolors.to_rgba(c, alpha=0.6) for c in palette]

    font_properties = font_manager.FontProperties(family="Times New Roman", style="normal", )
    plt.rcParams['font.family'] = 'Times New Roman'
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    all_x, all_y = [], []
    for i, item in enumerate(tqdm(sim_list)):
        assert isinstance(item, list)
        all_x += [data_name_list[i]] * len(item)
        all_y += item
    
    print("began plotting")
    sns.violinplot(x=all_x, y=all_y, palette=palette, linewidth=2, fill=True, ax=ax)
    for violin in ax.collections:
        violin.set_alpha(0.6)
    
    ax.set_ylabel('Cosine Similarity', fontsize=24, fontproperties=font_properties)
    ax.set_xlabel('Model', fontsize=24, fontproperties=font_properties)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    
    fig.tight_layout()
    os.makedirs("./figs", exist_ok=True)
    file_name = f'./figs/simscore_violin_vertical_all{datasource}.png'
    print("Saving figures %s" % file_name)
    fig.savefig(file_name)
    plt.close(fig)


def visualize_codebook_diversity(used=False, datasource="casp14"):

    NAMES = {
        "esm3": "ESM3",
        "foldseek": "FoldSeek",
        "protokens": "ProTokens",
        "ourpretrained_VanillaVQ": "VanillaVQ",
        "ourpretrained_AminoAseed": "AminoAseed",
    }

    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if not used:
        data_file_list = glob(os.path.join(dir_name, f"./tmp_simscore_dist/*cos*"))
        data_file_list = sorted(data_file_list)
        
        list_of_sims, list_of_names = [], []
        for file in data_file_list:
            if datasource not in file:
                continue
            sim_score = torch.load(file)
            #data_name = file.split("/")[-1].split("_")[-2]#.replace("simscore_", "")
            data_name = file.split("/")[-1].replace("simscore_cos_", "").replace("_casp14", "").replace("_cameo", "")
            if data_name not in NAMES:
                continue
            data_name = NAMES[data_name]
            
            sim_score = sim_score.reshape(-1).tolist()
            list_of_sims.append(sim_score)
            list_of_names.append(data_name)
    else:
        data_file_list = glob(os.path.join(dir_name, f"./tmp_simscore_used_dist/*cos*"))
        data_file_list = sorted(data_file_list)

        list_of_sims, list_of_names = [], []
        for file in data_file_list:
            if datasource not in file:
                continue
            used_hist, sim_score = torch.load(file)
            total = used_hist.sum().item()
            total_min = (torch.tensor([x for x in used_hist if x != 0]) / total).min().item()
            for i in range(len(used_hist)):
                
                if used_hist[i] == 0:
                    continue
                used_hist[i] = (used_hist[i] / total) / total_min
            used_hist = torch.round(used_hist).cpu().int().tolist()
            
            new_sim_score = []
            for idx, val in enumerate(tqdm(used_hist)):
                    for jidx, jval in enumerate(used_hist):
                        new_sim_score += [sim_score[idx][jidx]] * (val * jval)

            
            data_name = file.split("/")[-1].replace("simscore_cos_", "").replace("_casp14", "").replace("_cameo", "")
            if data_name not in NAMES:
                continue
            data_name = NAMES[data_name]
            list_of_sims.append(new_sim_score)
            list_of_names.append(data_name)
    
    print("finished processing ...")
    
    names_order = ["FoldSeek", "ProTokens", "ESM3", "VanillaVQ", "AminoAseed"]
    new_list_of_sims = []
    for x in names_order:
        for idx, y in enumerate(list_of_names):
            if y == x:
                new_list_of_sims.append(list_of_sims[idx])
                break
    visualize_simscore_violin(new_list_of_sims, names_order, "_" + datasource)


if __name__ == "__main__":
    
    visualize_codebook_diversity()