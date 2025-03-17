# https://www.youtube.com/watch?v=O2Cw82YR5Bo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ecommerce.csv")
# Makes it so it shows the full df
pd.set_option("display.max_columns", None)
pd.set_option("display.width",None)
df.head() 

df.info() # Gets us the column name and other general info

df.describe() # this is like summary() in R


# EDA - Exploratory Data Analysis
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df)
plt.show()