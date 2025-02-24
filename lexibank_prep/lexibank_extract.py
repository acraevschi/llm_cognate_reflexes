import pandas as pd
import requests
import os


def find_relevant(folder="lexibank"):
    file_list = os.listdir(folder)
    tsv_files = [file for file in file_list if file.endswith(".tsv")]

    files_to_download = []
    for file_name in tsv_files:
        df = pd.read_csv(f"{folder}/{file_name}", sep="\t")
        filtered_df = df[(df["Organization"] == "lexibank") & (df["CogCore"].notna())]
        files_to_download.extend(filtered_df["ID"].to_list())
    return files_to_download


def download_relevant(folder="lexibank"):
    files_to_download = find_relevant(folder)
    lexibank_files = ["languages", "cognates", "forms"]
    no_success = dict()
    for file_type in lexibank_files:
        for file_id in files_to_download:
            url = f"https://raw.githubusercontent.com/lexibank/{file_id}/main/cldf/{file_type}.csv"
            r = requests.get(url)
            if r.status_code == 404:
                url = f"https://raw.githubusercontent.com/lexibank/{file_id}/master/cldf/{file_type}.csv"
                r = requests.get(url)
                if r.status_code == 404:
                    no_success[file_id] = "No repo found"
                    continue

                df = pd.read_csv(url, encoding="utf-8")
                folder_path = os.path.join("lexibank", file_id)
                os.makedirs(folder_path, exist_ok=True)

                df.to_csv(
                    f"{folder_path}/{file_type}.csv", index=False, encoding="utf-8"
                )

        if no_success:
            print("These files were not successfully downloaded:")
            for file_id, reason in no_success.items():
                print(f"{file_id}: {reason}")


if __name__ == "__main__":
    download_relevant()
