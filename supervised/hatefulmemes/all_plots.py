from plots import *
import matplotlib.pyplot as plt


def all_plots(df_list):
fig, axes = plt.subplots(nrows=2, ncols=2)

df1.plot(ax=axes[0,0])
df2.plot(ax=axes[0,1])