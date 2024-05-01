import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Assuming your data is stored in a CSV file named 'data.csv'
data = pd.read_csv('3D_plot_data.csv', names=['Var1', 'Var2', 'Dependent'])

# Fit the ANOVA model
model = ols('Dependent ~ C(Var1) + C(Var2) + C(Var1):C(Var2)', data=data).fit()

# Perform ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
