import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')


# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6
    df_cat['total'] = 1
    #df_cat['total'] = df_cat.groupby(['cardio', 'variable', 'value'], as_index = False).count()
    #print(df_cat)
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).count()
    #print(df_cat)
    #df_cat = None

    # 7



    # 8
    fig = sns.catplot(data=df_cat, x='variable', y='total', col='cardio', hue='value', kind='bar').figure
    #fig = sns.catplot(data=df_cat, x='variable', col='cardio', hue='value', kind='count').set_axis_labels('variable', 'total')
    #plt.show()

    # 9
    fig.savefig('catplot.png')
    return fig
#draw_cat_plot()

# 10
def draw_heat_map():
    # 11
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) & (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(corr)



    # 14
    fig, ax = plt.subplots(figsize=(9, 9))

    # 15
    sns.heatmap(corr, mask=mask, square=True, annot=True, linewidths=.5, center=0, vmax=.32, cbar_kws={"shrink": .5, "ticks": np.arange(-0.16, 0.32, 0.08)}, fmt=".1f")
    #plt.show()
    # 16
    fig.savefig('heatmap.png')
    return fig
#draw_heat_map()