import pandas as pd


df = pd.read_csv("cleaned_data.csv")

df['Decision'] = df['Decision'].apply(lambda x : 1 if str(x).lower() == "fit" else 0)



df.to_csv("converted_cleaned_data.csv" , index=False)