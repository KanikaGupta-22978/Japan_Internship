import pandas as pd
from datetime import timedelta
import os

# Mapping from user ID to FonLog user name
user_map = {
    "Takasu01": "高須１番",
    "Takasu02": "高須２番",
    "Takasu03": "高須３番",
    "Takasu04": "高須４番",
    "Takasu05": "高須５番",
    "Takasu06": "高須６番",
    "Takasu07": "高須７番",
    "Takasu08": "高須８番",
    "Takasu09": "高須９番",
    "Takasu10": "高須10番"
}

# Paths
base_input_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\preprocessed"
fonlog_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\fonlog_qol_ready.csv"
output_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\data\combined"

# Load FonLog once
fonlog = pd.read_csv(fonlog_path)
fonlog["Started At"] = pd.to_datetime(fonlog["Started At"])

for user_id, fonlog_name in user_map.items():
    try:
        print(f"▶ Processing {user_id}...")

        # File paths
        garmin_file = os.path.join(base_input_path, f"{user_id}_garmin_preprocessed.xlsx")
        save_file = os.path.join(output_path, f"{user_id}_combined.csv")

        # Load Garmin data
        garmin = pd.read_excel(garmin_file, sheet_name='garmin', engine='openpyxl')
        garmin["Timestamp"] = pd.to_datetime(garmin["Timestamp"])

        # Filter FonLog entries for this user
        user_fonlog = fonlog[fonlog["User Name"] == fonlog_name]

        # Create 15-minute intervals in Garmin data
        garmin["interval_start"] = garmin["Timestamp"].dt.floor("15min")
        garmin["interval_end"] = garmin["interval_start"] + timedelta(minutes=15)
        garmin["bad_qol"] = pd.NA

        # Filter FonLog entries within the Garmin data time range
        valid_fonlog = user_fonlog[
            (user_fonlog["Started At"] >= garmin["interval_start"].min()) &
            (user_fonlog["Started At"] < garmin["interval_end"].max())
        ]

        # Match FonLog entries to Garmin intervals
        for _, f_row in valid_fonlog.iterrows():
            mask = (garmin["interval_start"] <= f_row["Started At"]) & (f_row["Started At"] < garmin["interval_end"])
            garmin.loc[mask, "bad_qol"] = f_row["bad_qol"]

        # Forward fill bad_qol values
        garmin = garmin.sort_values("Timestamp").reset_index(drop=True)
        garmin["bad_qol"] = garmin["bad_qol"].ffill()
        garmin = garmin[garmin["bad_qol"].notna()].reset_index(drop=True)

        # Drop helper columns
        garmin = garmin.drop(columns=["interval_start", "interval_end"])

        # Save merged data
        garmin.to_csv(save_file, index=False)
        print(f"✅ Saved: {save_file} | Matched entries: {len(valid_fonlog)}")

    except Exception as e:
        print(f"❌ Failed to process {user_id}: {e}")
