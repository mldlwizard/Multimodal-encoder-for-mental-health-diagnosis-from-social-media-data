import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors



def plot_loss_accuracy(data, hyperparameters, save_path):
    # Create subplots
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(15, 10))

    # Plot train and val loss
    sns.lineplot(x='epoch', y='train_loss', data=data, label='train', ax=axs[0][0])
    sns.lineplot(x='epoch', y='val_loss', data=data, label='val', ax=axs[0][0])
    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Loss')
    axs[0][0].set_title('Train and Val Loss')

    # Plot train and val accuracy
    sns.lineplot(x='epoch', y='train_acc', data=data, label='train', ax=axs[0][1])
    sns.lineplot(x='epoch', y='val_acc', data=data, label='val', ax=axs[0][1])
    axs[0][1].set_xlabel('Epochs')
    axs[0][1].set_ylabel('Accuracy')
    axs[0][1].set_title('Train and Val Accuracy')

    # Plot train and val precision
    sns.lineplot(x='epoch', y='train_precision', data=data, label='train', ax=axs[0][2])
    sns.lineplot(x='epoch', y='val_precision', data=data, label='val', ax=axs[0][2])
    axs[0][2].set_xlabel('Epochs')
    axs[0][2].set_ylabel('Precision')
    axs[0][2].set_title('Train and Val Precision')

    # Plot train and val recall
    sns.lineplot(x='epoch', y='train_recall', data=data, label='train', ax=axs[1][0])
    sns.lineplot(x='epoch', y='val_recall', data=data, label='val', ax=axs[1][0])
    axs[1][0].set_xlabel('Epochs')
    axs[1][0].set_ylabel('Recall')
    axs[1][0].set_title('Train and Val Recall')

    # Plot train and val F1 score
    sns.lineplot(x='epoch', y='train_f1score', data=data, label='train', ax=axs[1][1])
    sns.lineplot(x='epoch', y='val_f1score', data=data, label='val', ax=axs[1][1])
    axs[1][1].set_xlabel('Epochs')
    axs[1][1].set_ylabel('F1 Score')
    axs[1][1].set_title('Train and Val F1 Score')

    # Remove empty plot
    axs[1][2].remove()

    # Add text below the plot
    # plt.subplots_adjust(bottom=0.2, hspace=0.4)
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # axs[1][1].text(0.5, -0.3, str(hyperparameters), transform=axs[1][1].transAxes, fontsize=6,
    #                verticalalignment='top', horizontalalignment='center', bbox=props)

    # Display the plot
    plt.show()
    plt.savefig(save_path)




def combined_plot(df_list, name_list):

    metrics_df = pd.DataFrame(columns = df_list[0].columns)
    for i in range(len(df_list)):
        metrics_df.loc[i,:] = df_list[i].loc[df_list[i].shape[0]-1,:]

    metrics_df["model_name"] = name_list

    return metrics_df


def plot_metrics(df, save_path):
     # Create a dictionary to map each model name to a unique integer
    model_names = df["model_name"].unique()
    model_dict = {name: i for i, name in enumerate(model_names)}

    # Set Seaborn style
    sns.set_style("white")
    sns.color_palette("colorblind")

    # Set up the subplots
    fig, axs = plt.subplots(5, 2, figsize=(30, 30), gridspec_kw={'wspace': 0.75, 'hspace': 0.75})
    axs = axs.flatten()

    # Set the colors for each subplot
    colors = sns.color_palette("colorblind", 5).as_hex()


    # Plot train_loss
    train_loss = (df["train_loss"].values)
    x = np.arange(len(train_loss))
    axs[0].bar(x, train_loss, color=colors[0])
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[0].set_ylabel("Train Loss")
    axs[0].set_title("Train Loss by Model")
    axs[0].margins(x=0.1)
    for i, v in enumerate(train_loss):
        axs[0].text(i, v+0.04, str(round(v, 3)), ha="center", va="top")

    # Plot val_loss
    val_loss = df["val_loss"].values
    axs[1].bar(x, val_loss, color=colors[0])
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[1].set_ylabel("Validation Loss")
    axs[1].set_title("Validation Loss by Model")
    axs[1].margins(x=0.1)
    for i, v in enumerate(val_loss):
        axs[1].text(i, v+0.04, str(round(v, 3)), ha="center", va="top")

    # Plot train_acc
    train_acc = df["train_acc"].values
    axs[2].bar(x, train_acc, color=colors[1])
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[2].set_ylabel("Train Accuracy")
    axs[2].set_title("Train Accuracy by Model")
    axs[2].margins(x=0.1)
    for i, v in enumerate(train_acc):
        axs[2].text(i, v+0.04, str(round(v, 3)), ha="center", va="top")

    # Plot val_acc
    val_acc = df["val_acc"].values
    axs[3].bar(x, val_acc, color=colors[1])
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[3].set_ylabel("Validation Accuracy")
    axs[3].set_title("Validation Accuracy by Model")
    axs[3].margins(x=0.1)
    for i, v in enumerate(val_acc):
        axs[3].text(i, v+0.04, str(round(v, 3)), ha="center", va="top")


    # Plot train_recall
    train_rec = df["train_recall"].values
    axs[4].bar(x, train_rec, color=colors[2])
    axs[4].set_xticks(x)
    axs[4].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[4].set_ylabel("Train Recall")
    axs[4].set_title("Train Recall by Model")
    axs[4].margins(x=0.1)
    for i, v in enumerate(train_rec):
        axs[4].text(i, v+0.02, str(round(v, 3)), ha="center", va="top")

    # Plot val_recall
    val_rec = df["val_recall"].values
    axs[5].bar(x, val_rec, color=colors[2])
    axs[5].set_xticks(x)
    axs[5].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[5].set_ylabel("Validation Recall")
    axs[5].set_title("Validation Recall by Model")
    axs[5].margins(x=0.1)
    for i, v in enumerate(val_rec):
        axs[5].text(i, v+0.02, str(round(v, 3)), ha="center", va="top")

    # Plot train_precision
    train_prec = df["train_precision"].values
    axs[6].bar(x, train_prec, color=colors[3])
    axs[6].set_xticks(x)
    axs[6].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[6].set_ylabel("Train Precision")
    axs[6].set_title("Train Precision by Model")
    axs[6].margins(x=0.1)
    for i, v in enumerate(train_prec):
        axs[6].text(i, v+0.02, str(round(v, 3)), ha="center", va="top")

    # Plot val_precision
    val_prec = df["val_precision"].values
    axs[7].bar(x, val_prec, color=colors[3])
    axs[7].set_xticks(x)
    axs[7].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[7].set_ylabel("Validation Precision")
    axs[7].set_title("Validation Precision by Model")
    axs[7].margins(x=0.1)
    for i, v in enumerate(val_prec):
        axs[7].text(i, v+0.02, str(round(v, 3)), ha="center", va="top")

    # Plot train_f1score
    train_f1 = df["train_f1score"].values
    axs[8].bar(x, train_f1, color=colors[4])
    axs[8].set_xticks(x)
    axs[8].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[8].set_ylabel("Train F1 Score")
    axs[8].set_title("Train F1 Score by Model")
    axs[8].margins(x=0.1)
    for i, v in enumerate(train_f1):
        axs[8].text(i, v+0.02, str(round(v, 3)), ha="center", va="top")

    # Plot val_f1score
    val_f1 = df["val_f1score"].values
    axs[9].bar(x, val_f1, color=colors[4])
    axs[9].set_xticks(x)
    axs[9].set_xticklabels(df["model_name"].values, rotation=45, ha="right")
    axs[9].set_ylabel("Validation F1 Score")
    axs[9].set_title("Validation F1 Score by Model")
    axs[9].margins(x=0.1)
    for i, v in enumerate(val_f1):
        axs[9].text(i, v+0.02, str(round(v, 3)), ha="center", va="top")
        
    # Adjust the subplot spacing
    # plt.subplots_adjust(top=1)
    # Show the plot
    plt.show()

    plt.savefig(save_path)


def plot_hp_comparison(hp_tuning_dir, metric):

    for file_name in os.listdir(hp_tuning_dir + "/metrics/"):

        df = pd.read_csv(hp_tuning_dir + "/metrics/" + file_name)

        filtered_df = df[df[metric] == df[metric].max()].tail(1)

        try:
            final_df = pd.concat([final_df, filtered_df], axis=0)
        except:
            final_df = filtered_df.copy()

    final_df = final_df.reset_index(drop=True)
    final_df["model_name"] = final_df.index
    # print(final_df)
        
    plot_metrics(final_df, hp_tuning_dir + "comparison_hp_" + metric + ".png")


def plot_model_comparison(model_comp_dir, metric):

    for file_name in os.listdir(model_comp_dir + "/metrics/"):

        df = pd.read_csv(model_comp_dir + "/metrics/" + file_name)

        filtered_df = df[df[metric] == df[metric].max()].tail(1)
        filtered_df["model_name"] = file_name.split(".csv")[0]

        try:
            final_df = pd.concat([final_df, filtered_df], axis=0)
        except:
            final_df = filtered_df.copy()

    final_df = final_df.reset_index(drop=True)
    # final_df["model_name"] = final_df.index
    # print(final_df)
        
    plot_metrics(final_df, model_comp_dir + "comparison_models_" + metric + ".png")


model_comp_dir = '/home/nippani.a/Silvio/Multimodal/results/model_comps/'

plot_model_comparison(model_comp_dir, "val_acc")
plot_model_comparison(model_comp_dir, "val_f1score")


hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_unimodal_img/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")

hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_unimodal_txt/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")

hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/early_low_rank_fusion/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")


hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_unimodal_img/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")


hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_unimodal_txt/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")


hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_low_rank_fusion/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")

hp_tuning_dir = '/home/nippani.a/Silvio/Multimodal/results/hp_tuning/late_fusion/'

plot_hp_comparison(hp_tuning_dir, "val_acc")
plot_hp_comparison(hp_tuning_dir, "val_f1score")


# print(df_list)
# print(name_list)