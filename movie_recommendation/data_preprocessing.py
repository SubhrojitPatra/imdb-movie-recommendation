#import requried libaries
import pandas as pd
import numpy as np
from pathlib import Path

#Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
#load the imdb dataset
CSV_PATH = PROJECT_ROOT / "IMDB-Movie-Data.csv"
df = pd.read_csv(CSV_PATH)
#Display basic information
print("Dataset loaded successfully!")
print(df.head())
print("Dataset Shape:",df.shape)
df.info()
'''
# Check for missing values
print(df.isnull().sum())
'''
#check duplicate records
print("Duplicated records:",df.duplicated().sum())
#check duplicated movie titles
print("Duplicate movie titles:",df["Title"].duplicated().sum())

# Rename columns
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("(", "", regex=False)
df.columns = df.columns.str.replace(")", "", regex=False)

print("Updated columns:")
print(df.columns.tolist())

#Combine important movie information
df["combined_features"] = (
    df["genre"] + " " + 
    df["description"] + " " + 
    df["director"] + " " +
    df["actors"]
)
print(df[["title", "combined_features"]].head())

#save processed data to a new CSV file
OUTPUT_PATH = PROJECT_ROOT / "movie_recommendation" / "processed_movie_data.csv"
df.to_csv(OUTPUT_PATH, index=False)
print("Processed dataset saved successfully!")