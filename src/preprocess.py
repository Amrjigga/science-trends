# Text preprocessing module for arXiv papers
# Cleans and normalizes titles and abstracts for downstream analysis

import re, pandas as pd, yaml, pathlib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Clean and normalize text by removing LaTeX, URLs, and special characters
def clean(txt:str)->str:
    txt = txt or ""
    txt = txt.lower()
    txt = re.sub(r"\$.*?\$", " ", txt)                 
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)       
    txt = re.sub(r"[^a-z0-9\s\-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# Main preprocessing function that loads, cleans, and saves paper data
def run(raw_json:str, out_parquet:str, text_fields:list):
    # Changed from CSV to JSON since we have arxiv-metadata-oai-snapshot.json
    df = pd.read_json(raw_json, lines=True)
    tf = [c for c in text_fields if c in df.columns]
    df["text"] = df[tf].astype(str).agg(" ".join, axis=1).map(clean)
    # keep date if present
    for c in ["update_date","versions","created","submitted"]:
        if c in df.columns: 
            df["date"] = pd.to_datetime(df[c], errors="coerce").fillna(pd.NaT)
            break
    pathlib.Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df[["id","categories","text","date"]].to_parquet(out_parquet, index=False)

if __name__ == "__main__":
    print("Starting preprocessing...")
    cfg = yaml.safe_load(open("src/config.yaml"))
    run("data/raw/arxiv-metadata-oai-snapshot.json",
        "data/interim/clean.parquet",
        cfg["text_fields"])
    print("Preprocessing complete!")
