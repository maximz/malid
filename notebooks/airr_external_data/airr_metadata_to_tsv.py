import sys, json
import pandas as pd

# pip install flatten_json
from flatten_json import flatten

data = json.loads(sys.stdin.read())

dic_flattened = (flatten(d, ".") for d in data["Repertoire"])
df = pd.DataFrame(dic_flattened)

sys.stdout.write(df.to_csv(index=None, sep="\t"))
