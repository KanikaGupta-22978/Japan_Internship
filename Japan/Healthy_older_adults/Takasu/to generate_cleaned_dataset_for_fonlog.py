import pandas as pd
import re

# Step 1: Load the activity targets CSV
# ステップ1：アクティビティCSVを読み込む
activity_file = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\activity_targets-20250519.csv"
df = pd.read_csv(activity_file, encoding="cp932")

# Step 2: Keep only rows where 'Records' is not empty
# ステップ2：Records列が空でない行のみ処理
df_records = df[['Activity ID', 'Records']].dropna()

# Step 3: Extract Q&A pairs from Records column
# ステップ3：Records列から【質問】回答ペアを抽出
def extract_qa(row):
    activity_id = row["Activity ID"]
    text = row["Records"]
    qa_pairs = re.findall(r'【(.*?)】([^,【】]+)', text)
    return pd.DataFrame([
        {
            "Activity ID": activity_id,
            "Question": q.strip(),
            "Answer": a.strip()
        } for q, a in qa_pairs
    ])

qa_df = pd.concat([extract_qa(row) for _, row in df_records.iterrows()], ignore_index=True)

# Step 4: Map answers to QoL levels
# ステップ4：回答をQOLレベルに変換
def map_to_qol_level(answer):
    if answer in ["ある", "とてもある", "頻繁にある"]:
        return "Moderate"
    elif answer in ["少しある"]:
        return "Minor"
    elif answer in ["ない"]:
        return "None"
    else:
        return "Unknown"

qa_df["EQ5D_Level"] = qa_df["Answer"].apply(map_to_qol_level)

# Step 5: Merge additional metadata from original CSV
# ステップ5：元データから追加情報をマージ
columns_to_add = [
    "Activity ID",
    "Activity Type ID",
    "Activity Type Group",
    "Target ID",
    "Target",
    "Started",
    "Finished"
]
metadata_df = df[columns_to_add].drop_duplicates()

# Merge QA data with metadata
# QAデータにメタデータを付加
qa_merged = pd.merge(qa_df, metadata_df, on="Activity ID", how="left")

# Step 6: Rename columns to match FonLog code expectations
# ステップ6：列名をFonLog用にリネーム
qa_merged.rename(columns={
    "Activity Type ID": "Activity Type ID",
    "Activity Type Group": "Group Name",
    "Target ID": "User ID",
    "Started": "Started At",
    "Target": "User Name",
    "Finished": "Finished At"
}, inplace=True)

# Step 7: Save to CSV for FonLog preprocessing
# ステップ7：FonLogコード用にCSV保存
output_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\processed_fonlog_input_with_metadata.csv"
qa_merged.to_csv(output_path, index=False, encoding="cp932")

print("✅ Saved to:", output_path)
