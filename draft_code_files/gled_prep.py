import pandas as pd

gled = pd.read_csv('gled.tsv', sep='\t', encoding='utf-8')

all_langs = list(set(gled["FAMILY"]))
all_langs = list(map(lambda x: x.replace(' ', '_').replace("-", "").lower(), all_langs))
