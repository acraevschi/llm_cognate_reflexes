import pandas as pd


def check_glotto_coverage(folder):
    # Quickly check the coverage of Glottocodes and return a string indicating which way to go
    lang_df = pd.read_csv(f"{folder}/languages.csv")
    try:
        non_missing_percentage = lang_df["Glottocode"].notna().mean() * 100
        if non_missing_percentage >= 90:
            return "Glotto_only"
        elif non_missing_percentage >= 70:
            return "Glotto_and_non-Glotto"
        else:
            return "Non-Glotto_only"
    except:
        return "Non-Glotto_only"
