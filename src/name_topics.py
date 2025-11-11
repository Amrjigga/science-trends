# Topic naming module for converting TF-IDF terms into human readable topic names. because it was using ids before
# Uses pattern matching rules to identify common research areas and themes

import pandas as pd, re

#  map key terms to names
RULES = [
    (r"\b(llms?|language models?)\b", "Large Language Models"),
    (r"\bgraph\b|\bgnn(s)?\b|\bnetworks?\b.*graph", "Graph Neural Networks"),
    (r"\bdeep\b|\bneural\b", "Deep Learning (General)"),
    (r"\bquantum (information|entanglement|qubit|mechanics)\b|\bqubit\b", "Quantum Information"),
    (r"\bspin\b|\bsuperconduct(ing|ivity)\b|\blattice\b|\bcondensed\b", "Condensed-Matter Physics"),
    (r"\bstring\b|\bbrane\b|\bgauge\b|qft|\bads/cft\b", "String Theory / QFT"),
    (r"\bgalax(y|ies)\b|\bstellar\b|\bredshift\b|\bradio\b", "Astrophysics"),
    (r"\bneutrino\b|\bqcd\b|\bhiggs\b|\bcollision(s)?\b|\bgev\b", "High-Energy Physics"),
    (r"\balgebra(s)?\b|\blie\b|\bcohomology\b|\bmanifolds?\b|\btheorem\b", "Pure Mathematics (Algebra/Topology)"),
    (r"\bequation(s)?\b|\bgravity\b|\bdimensional\b|\bboundary\b", "Mathematical Physics"),
]

# Convert TF-IDF terms to meaningful topic names using pattern matching
def guess_name(terms: str) -> str:
    low = " " + terms.lower() + " "
    for pat, name in RULES:
        if re.search(pat, low):
            return name
    
    parts = [t.strip() for t in terms.split(",") if len(t.strip())>=3][:3]
    return " / ".join(w.capitalize() for w in parts)

if __name__ == "__main__":
    print("Converting cluster terms to topic names...")
    labels_df = pd.read_csv("outputs/tables/cluster_labels.csv", index_col=0)
    # labels_df has cluster IDs as index and terms in column '0'
    out = []
    for cid, row in labels_df.iterrows():
        terms = row.iloc[0]  # Get the terms from the first (and only) column
        if pd.isna(terms):  
            continue
        name = guess_name(terms)
        out.append({"cluster": int(cid), "name": name, "terms": terms})
        print(f"Cluster {cid}: {name}")
    df = pd.DataFrame(out).sort_values("cluster")
    df.to_csv("outputs/tables/topic_names.csv", index=False)
    print("Wrote outputs/tables/topic_names.csv")
