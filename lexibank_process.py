import pandas as pd
from lexibank_extract import find_relevant
import json

def long_to_wide(folders):
    fam_counts = dict()
    for folder in folders:
        forms = pd.read_csv(f'lexibank/{folder}/forms.csv')
        langs = pd.read_csv(f'lexibank/{folder}/languages.csv')
        # keep only rows with at least two cognates
        cognate_col = "Cognacy"
        if forms["Cognacy"].isnull().all():
            cognate_col = "Parameter_ID"
        forms = forms[forms[cognate_col].duplicated(keep=False)]
        forms["Family"] = forms["Language_ID"].map(langs.set_index("ID")["Family"])
        forms["Family"] = forms["Family"].str.replace('[^\w\s]', '').str.replace(' ', '_').str.replace("-", "").str.lower()
        counts_dict = forms["Family"].value_counts().to_dict()
        for key, value in counts_dict.items():
            if key in fam_counts:
                fam_counts[key] += value
            else:
                fam_counts[key] = value
        if cognate_col == "Cognacy":
            forms['Cognate_ID'] = forms['Cognacy'].astype(str) + '_' + forms['Parameter_ID']
        else:
            forms['Cognate_ID'] = forms['Parameter_ID']
        wide_df = forms.pivot_table(index='Cognate_ID', columns='Language_ID', values='Segments', aggfunc='first')
        wide_df.reset_index(inplace=True)
        wide_df.fillna('', inplace=True)
        wide_df.to_csv(f'lexibank/{folder}/wide_df.tsv', index=False, sep="\t", encoding='utf-8')

if __name__ == "__main__":
    folders = find_relevant()
    fam_counts = long_to_wide(folders)

