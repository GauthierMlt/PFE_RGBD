import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.stats import t
import os
from scipy.stats import pearsonr
import seaborn as sns

def reg_coef(x,y,label=None,color=None,**kwargs):
    ax = plt.gca()
    r,p = pearsonr(x,y)
    ax.annotate('r = {:.2f}'.format(r), xy=(0.5,0.5), xycoords='axes fraction', ha='center')
    ax.set_axis_off()

# Setting working directory to the path of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv("Donnees_sujets.csv")
data=data.dropna()
data=data.dropna(axis=0)

## BOXPLOTS 
# plt.figure()
# plt.subplot(2,1,1)
# data.boxplot(column=["Epaisseur genou (cm)", 
#                      "Epaisseur abdomen au niveau ombilical de profil (cm)", 
#                      "Epaisseur abdomen de face au niveau ombilical (cm)"])
# plt.subplot(2,1,2)
# data.boxplot(column=["Epaisseur 1ere phalange index (cm)"])

def plotData(X_name: str, Y_name: str ="IMC", regression: bool=True):
    X = data[X_name]
    Y = data[Y_name]
    
    if regression:
        rl = linregress(X, Y)

        tinv = lambda p, df: abs(t.ppf(p/2, df))
        ts = tinv(0.05, len(X)-2)

        print(f"slope (95%): {rl.slope:.6f} +/- {ts*rl.stderr:.6f}")

        print(f"intercept (95%): {rl.intercept:.6f}" \
        f" +/- {ts*rl.intercept_stderr:.6f}")
    
    plt.figure()
    plt.axes().grid()
    plt.scatter(X, Y)
    
    if regression:
        plt.plot(X, rl.intercept + rl.slope*X, 'r', 
                label=f'R: {rl.rvalue:.4f} | p-val: {rl.pvalue:.2e}')
        plt.legend()
        
    plt.xlabel(X_name)
    plt.ylabel(Y_name)


## SCATTER PLOTS
# plotData("Epaisseur abdomen au niveau ombilical de profil (cm)")
# plotData("Epaisseur abdomen de face au niveau ombilical (cm)")
# plotData("Epaisseur 1ere phalange index (cm)", regression=False)
# plt.show()

# SCATTER MATRIX
data_measures = data[[
    "IMC", 
    "Epaisseur abdomen au niveau ombilical de profil (cm)",
    "Epaisseur abdomen de face au niveau ombilical (cm)",
    "Epaisseur 1ere phalange index (cm)"]]

data_measures.rename(columns={
   "Epaisseur abdomen au niveau ombilical de profil (cm)" : "Epaisseur abdomen profil",
   "Epaisseur abdomen de face au niveau ombilical (cm)": "Epaisseur abdomen face",
   "Epaisseur 1ere phalange index (cm)": "Epaisseur index"
}, inplace=True)

# pd.plotting.scatter_matrix(data_measures,figsize=(20,20),grid=True)
g = sns.PairGrid(data_measures)
g.map_diag(sns.distplot)
g.map_lower(sns.regplot)
g.map_upper(reg_coef)
plt.show()