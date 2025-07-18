import pandas as pd

# ✅ Step 1: Load the raw CSV
input_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\processed_fonlog_input_with_metadata.csv"
df = pd.read_csv(input_path, encoding="cp932")

# ✅ Step 2: Map QoL questions to English
qol_dictionary = {
    '"今日の"あなたの「移動の程度」を最もよく表した項目を１つ選択してください。': "Degree of mobility",
    '"今日の"あなたの「身の回りの管理」を最もよく表した項目を１つ選択してください。': "Personal care",
    '"今日の"あなたの「ふだんの行動」を最もよく表した項目を１つ選択してください。（例：仕事、勉強、家事、家族･余暇活動）': "Daily activities",
    '"今日の"あなたの「痛み / 不快感」を最もよく表した項目を１つ選択してください。': "Pain / discomfort",
    '"今日の"あなたの「不安 / ふさぎ込み」を最もよく表した項目を１つ選択してください。': "Anxiety / distraction"
}

df = df[df["Question"].isin(qol_dictionary.keys())].copy()
df["QoL_Item"] = df["Question"].map(qol_dictionary)

# ✅ Step 3: Convert answers to 1–5 scale
def convert_answer_to_score(ans):
    if pd.isna(ans):
        return None
    elif any(x in ans for x in ["問題はない", "痛みや不快感はない", "不安でもふさぎ込んでもいない"]):
        return 1
    elif "少し" in ans:
        return 2
    elif "中程度の" in ans:
        return 3
    elif "かなり" in ans:
        return 4
    elif any(x in ans for x in ["できない", "極度の"]):
        return 5
    else:
        return None

df["Score"] = df["Answer"].apply(convert_answer_to_score)

# ✅ Step 4: Forward fill long-format scores
df.sort_values(by=["Activity ID", "QoL_Item"], inplace=True)
df["Score"] = df.groupby("QoL_Item")["Score"].ffill()

# ✅ Step 5: Pivot to wide format
qol_wide = df.pivot_table(index="Activity ID", columns="QoL_Item", values="Score").reset_index()
qol_wide.columns.name = None

# ✅ Step 6: Forward fill again for any remaining missing values
qol_wide.fillna(method="ffill", inplace=True)

# ✅ Step 7: Label bad QoL
qol_items = list(qol_dictionary.values())
qol_wide["bad_qol"] = qol_wide[qol_items].apply(lambda row: 1 if any(row >= 2) else 0, axis=1)

# ✅ Step 8: Translate User Name to English
name_translation = {
    "槇戸 悠": "Yu_Makido",
    "高須１番": "Takasu01",
    "高須２番": "Takasu02",
    "高須３番": "Takasu03",
    "高須４番": "Takasu04",
    "高須５番": "Takasu05",
    "高須６番": "Takasu06",
    "高須７番": "Takasu07",
    "高須８番": "Takasu08",
    "高須９番": "Takasu09",
    "高須10番": "Takasu10"
}

df["User Name (English)"] = df["User Name"].map(name_translation)

# ✅ Step 9: Merge metadata and save
meta_cols = ["Activity ID", "Started At", "Finished At", "User ID", "User Name", "User Name (English)"]
meta_df = df[meta_cols].drop_duplicates()
final_df = pd.merge(qol_wide, meta_df, on="Activity ID", how="left")

# ✅ Step 10: Save final output
output_path = r"C:\Users\ftska\OneDrive\Desktop\Japan Internship'25\Macky\Takasu\fonlog_qol_ready.csv"
final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("✅ All missing QoL values filled. Final result saved to:", output_path)
