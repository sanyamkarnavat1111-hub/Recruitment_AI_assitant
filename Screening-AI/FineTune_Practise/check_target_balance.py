import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV
df = pd.read_csv("converted_cleaned_data.csv")
# Optional: print raw counts
print(df["Decision"].value_counts())