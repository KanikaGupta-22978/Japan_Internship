import pandas as pd
from pathlib import Path

# --- File paths ---
csv_path = Path(r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined\all_combined_combined.csv")
xlsx_dir = Path(r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\Macky_shared_combined")
output_path = csv_path.parent / "final_merged_combined.csv"

# --- Load the CSV file ---
csv_df = pd.read_csv(csv_path)
csv_df["source"] = "csv_combined"

# --- Load all 10 .xlsx files ---
xlsx_dfs = []
for i in range(1, 11):
    file = xlsx_dir / f"Takasu{i:02d}_combined.xlsx"
    if file.exists():
        df = pd.read_excel(file, engine="openpyxl")
        df["user_id"] = f"Takasu{i:02d}"
        df["source"] = "xlsx_shared"
        xlsx_dfs.append(df)
    else:
        print(f"⚠️ Missing file: {file}")

# --- Combine Excel files ---
xlsx_df = pd.concat(xlsx_dfs, ignore_index=True)

# --- Merge CSV + Excel datasets ---
merged_df = pd.concat([csv_df, xlsx_df], ignore_index=True)

# --- Save the final merged file ---
merged_df.to_csv(output_path, index=False)
print(f"✅ Final merged file saved to:\n{output_path}")
