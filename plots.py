import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.stats import t

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
plotData("Epaisseur abdomen au niveau ombilical de profil (cm)")
plotData("Epaisseur abdomen de face au niveau ombilical (cm)")
plotData("Epaisseur 1ere phalange index (cm)", regression=False)
plt.show()