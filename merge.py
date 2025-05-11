import pandas as pd
import os

# List of files to merge
files = [
    "Ll_2023.csv",  
    "Ll_2023.csv",  
    "Ll_2022.csv",  
    "Ll_2021.csv",  
    "Ll_2019.csv"
]

# Initialize an empty list to hold DataFrames
dataframes = []

# Process each file
for file in files:
    # Extract season
    if file == "Ll_2023.csv":
        season = 2023
    else:
        season = int(file.split("Ll_")[1].split(".")[0])

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)

    # Add the 'season' column
    df['season'] = season

    # Add the 'code' column with value 'ES1'
    df['code'] = 'FR1'

    # Drop rows with any null values
    

    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV file
output_file = "Ligue.csv"
merged_df.to_csv(output_file, index=False)

print(f"Merged file with cleaned data saved as {output_file}")
