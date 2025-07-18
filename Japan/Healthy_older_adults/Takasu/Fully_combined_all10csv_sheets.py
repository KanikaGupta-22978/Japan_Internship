import pandas as pd
import os
from pathlib import Path

# Folder path containing the 10 combined CSV files
folder = Path(r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined")

# Output file path
output_file = folder / "all_combined_combined.csv"

# Initialize list for all DataFrames
dfs = []

# Iterate over Takasu##_combined.csv files
for i in range(1, 11):
    filename = folder / f"Takasu{i:02d}_combined.csv"
    if filename.exists():
        df = pd.read_csv(filename)
        df["user_id"] = f"Takasu{i:02d}"
        dfs.append(df)
    else:
        print(f"⚠️ File not found: {filename}")

# Combine all and save
all_data = pd.concat(dfs, ignore_index=True)
all_data.to_csv(output_file, index=False)

print(f"✅ All files combined and saved to:\n{output_file}")
