import pandas as pd

# Load the merged CSV file
file_path = "laliga.csv"
df = pd.read_csv(file_path)

# Update the 'season' column
season_mapping = {
    22: 2022,
    21: 2021,
    20: 2020,
    19: 2019
}
df['season'] = df['season'].replace(season_mapping)

# Save the updated DataFrame back to the CSV file
df.to_csv(file_path, index=False)

print("Updated the 'season' column in the file.")
