import os 

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns


def draw_pairPlot(df, hue_colname):
    plt.clf()
    
    plt.rcParams["figure.figsize"] = [10, 10] # [10, 5] looks better though
    fig, ax = plt.subplots()

    sns.set_context("paper")
    # sns.set_style("whitegrid");
    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete 

    ax = sns.pairplot(df, hue=hue_colname, height=3)

    # plt.savefig(os.path.join(prefix, fig_name),
    # bbox_inches = 'tight', # get rid of the white space
    # pad_inches = 0, # ref: https://stackoverflow.com/questions/11837979/removing-white-space-around-a-saved-image-in-matplotlib/27227718
    # dpi=300)

    # plt.show()

    return 0 


def scatter_plot(x_, y_, h_, df, x_axis_label, y_axis_label):
    plt.rcParams["figure.figsize"] = [8, 6] # [10, 5] looks better though
    fig, ax = plt.subplots()

    sns.set_context("paper")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    # sns.set(style='whitegrid',)
    # sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})

    # sns.set_context("paper")

    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete 
    ax = sns.scatterplot(x=x_, y=y_, data=df, hue_order=["BL", "LLP", "HLP", "A"], hue=h_) # , palette="deep"

    # sns.set(style='whitegrid',)

    plt.legend(bbox_to_anchor=(1, 1.15), loc= "upper right", ncol=4, borderaxespad=0.)

    # ax = sns.scatterplot(x=x_, y=y_, data=df, hue=h_)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    return 0 


def pair_density_plot(x, y, legend_label, x_axis_label, y_axis_label):
    plt.rcParams["figure.figsize"] = [6, 4] # [10, 5] looks better though
    fig, ax = plt.subplots()

    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete
    sns.set_context("paper") 

    ax = sns.distplot(x, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, hist=False, label=legend_label[0])
    # ax = sns.distplot(y, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, hist=False, label = legend_label[1])

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    # plt.show()

    return 0


def draw_distPlot(df, x_, h_, x_axis_label, y_axis_label):
    plt.rcParams["figure.figsize"] = [6, 4] # [10, 5] looks better though
    fig, ax = plt.subplots()

    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete
    sns.set_context("paper")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")

    # ["BL", "LLP", "HLP", "A"] = ["Baseline", "Low level pain", "High level pain", "Affect"]
    ax = sns.kdeplot(data=df, x=x_, hue_order = ["BL", "LLP", "HLP", "A"], hue=h_, fill=True)

    # ax = sns.distplot(x, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, hist=False, label=legend_label[0])
    # ax = sns.distplot(y, kde = True, kde_kws = {'shade': True, 'linewidth': 3}, hist=False, label = legend_label[1])

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(bbox_to_anchor=(1, 1.15), loc= "upper right", ncol=4, borderaxespad=0.)

    # ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # plt.show()

    return 0 


def bar_plot(x_, y_, h_, df, x_axis_label, y_axis_label):
    plt.rcParams["figure.figsize"] = [6, 6] # [10, 5] looks better though
    fig, ax = plt.subplots()

    sns.set_context("paper")
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    # sns.set_style("ticks", {"xtick.major.size": 2, "ytick.major.size": 2})

    # plt.ylim(np.min(df[y_], np.max(df[y_]))) 
    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete 
    ax = sns.barplot(x=x_, y=y_, hue_order = ["KNN", "RF", "XGB"], hue=h_, data=df, palette="deep")

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    handles, labels = ax.get_legend_handles_labels()
    # print(handles, labels)
    plt.legend(bbox_to_anchor=(1, 1.15), loc= "upper right", ncol=3, borderaxespad=0.)
    # plt.legend(handles[0:2], ["Male", "Female"])

    return 0 


def box_plot(x_, y_, h_, df, x_axis_label, y_axis_label):

    plt.rcParams["figure.figsize"] = [6, 6] # [10, 5] looks better though
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()

    # sns.set_context("paper")

    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete 
    ax = sns.boxplot(x=x_, y=y_, hue_order = ["KNN", "RF", "XGB"], hue=h_, data=df, palette="deep") # muted, Set1, Set3, colorblind

    plt.legend(bbox_to_anchor=(1, 1.15), loc= "upper right", ncol=3, borderaxespad=0.)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    return 0 


def draw_linePlot(x_, y_, x_axis_label, y_axis_label):
    plt.rcParams["figure.figsize"] = [6, 4] # [10, 5] looks better though
    fig, ax = plt.subplots()

    sns.set_context("paper")
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots()
    # sns.palplot( sns.color_palette("deep", 10)) # change "Set3" ,"deep" to get new pallete 
    # ax = sns.barplot(x=x_, y=y_, hue=h_, data=df, palette="deep")

    ax = sns.lineplot(x=x_, y=y_)

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    return 0 


if __name__ == "__main__":
    # prefix = "/Users/taufeeq/rdevs/CIPAD/data/annotation/processed"
    # fig_prefix = "../figs"

    pass 