import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

ds = load_dataset("tatsu-lab/alpaca")

# remove duplicates
text_df = pd.DataFrame(ds)
print(text_df)

# convert df from dict to column
text_df = text_df['train'].apply(pd.Series)
print("Size of original dataset: " + str(text_df.size))

text_df = text_df.drop("text", axis = 1)

# remove lines where input OR output is empty
text_df = text_df.replace('', None).dropna(subset=['instruction', 'output'])
text_df = text_df.drop_duplicates()
print("Size of dataset after removing duplicates and empty input/output: " + str(text_df.size))

print(text_df)

# format to input - output pairst
# include instruction, context, response
# input = instruction column
# context = input column
# response = output column
text_df = text_df.rename(columns={'instruction': 'instruction', 'input': 'context', 'output': 'response'})
print(text_df)

# 80 - 20 random validation split
train_df, test_df = train_test_split(text_df, test_size=0.2, random_state=52)
print("Size of train dataset: " + str(train_df.size))
print("Size of test dataset: " + str(test_df.size))

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Saved preprocessed data to CSV files")