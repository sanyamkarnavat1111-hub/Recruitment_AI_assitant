
import pandas as pd
import re
import os
import string
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS



dataframe1 = pd.read_csv("Dataset/resume_screening_dataset.csv")
dataframe1 = dataframe1[['Resume' , 'Decision' , 'Job_Description']]


dataframe2 = pd.read_csv("Dataset/job_applicant_dataset.csv")
dataframe2 = dataframe2[['Resume' , 'Job Description' , 'Best Match']]

dataframe2.rename(columns={'Job Description' : "Job_Description" , "Best Match" : "Decision"} , inplace=True)


df = pd.concat([dataframe1 , dataframe2] , axis=0 , ignore_index=True).reset_index(drop=True)


df['Resume'] = df['Resume'].apply(lambda x : re.sub('Proficient in' , '' , x))
df['Resume'] = df['Resume'].apply(lambda x : re.sub("Here's a professional resume for" , '' , x) )


punctuations = string.punctuation
translator = str.maketrans('', '', string.punctuation)

def preprocess_text(text :str):
    # Remove punctuations
    text = text.translate(translator)

    # Tokenize the sentence
    tokenized_words = word_tokenize(text)


    cleaned_words = [word for word in tokenized_words if word not in STOPWORDS]


    return ' '.join(cleaned_words)


df[['Resume', 'Job_Description']] = df[['Resume', 'Job_Description']].map(preprocess_text)


# These are repeating a lot in data frame 1 

words_to_remove = ["here", "professional", "resume"]

# Create regex pattern: \b(word1|word2|word3)\b
pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'

df['Resume'] = df['Resume'].str.replace(pattern, '', regex=True, case=False)

# Optional: clean up extra spaces
df['Resume'] = df['Resume'].str.replace(r'\s+', ' ', regex=True).str.strip()


df['Decision'] = df['Decision'].apply(lambda x : "fit" if int(x) == 1 else "unfit")

os.makedirs("Dataset/Cleaned_Dataset" , exist_ok=True)

df.to_csv("Dataset/Cleaned_Dataset/cleaned_resume_screening.csv" , index=False)