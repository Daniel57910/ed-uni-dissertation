import json
import pandas as pd
meta = []
with open('column_desc.json') as f:
    columns = json.load(f)
    

for k, v in columns.items():
    meta.append({'feature name': k, 'feature description': v})

df = pd.DataFrame(meta)
print(df.to_latex(index=False))