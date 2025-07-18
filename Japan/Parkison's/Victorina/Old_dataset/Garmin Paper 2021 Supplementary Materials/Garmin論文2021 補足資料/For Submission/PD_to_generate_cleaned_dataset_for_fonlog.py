import pandas as pd
import re

# Step 1: Load the CSV file
activity_file = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Parkison's\Victorina\Old_dataset\Garmin Paper 2021 Supplementary Materials\Garmin論文2021 補足資料\For Submission\activity_targets-20250519.csv"
df = pd.read_csv(activity_file, encoding="cp932")

# Step 2: Drop rows with empty Records
df_records = df[['Activity ID', 'Records']].dropna()

# Step 3: Extract Q&A from Records column
def extract_record(row):
    activity_id = row["Activity ID"]
    text = row["Records"]
    qa_pairs = re.findall(r'【(.*?)】([^,【】]+)', text)
    return pd.DataFrame([
        {
            "activity_target.activity_id": activity_id,
            "record_type.name": q.strip(),
            "value": a.strip()
        } for q, a in qa_pairs
    ])

record_df = pd.concat([extract_record(row) for _, row in df_records.iterrows()], ignore_index=True)

# Step 4: Add and rename metadata columns
metadata_columns = [
    "Activity ID",
    "Activity Type ID",
    "Activity Type Group",
    "Target ID",
    "Started",
    "Finished"
]
metadata_df = df[metadata_columns].drop_duplicates()

# Rename for merge and final format
metadata_df.rename(columns={
    "Activity ID": "activity_target.activity_id",
    "Activity Type ID": "record_type.activity_type_id",
    "Activity Type Group": "activity_type_group.name",
    "Target ID": "activity_target.user_id",
    "Started": "activity.started_at",
    "Finished": "activity.finished_at"
}, inplace=True)

# Step 5: Merge metadata
record_full = pd.merge(record_df, metadata_df, on="activity_target.activity_id", how="left")

# Step 6: Export
output_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Parkison's\Victorina\Old_dataset\Garmin Paper 2021 Supplementary Materials\Garmin論文2021 補足資料\For Submission\PD_processed_fonlog_input_with_metadata.csv"
record_full.to_csv(output_path, index=False, encoding="cp932")

print("✅ Record format CSV saved to:", output_path)
