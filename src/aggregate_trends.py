# Trend aggregation module for analyzing cluster evolution over time
# Computes monthly shares, trend slopes, and classifies clusters as rising/declining/stable

import pandas as pd, numpy as np, yaml

# Calculate linear slope over the last n time periods
def slope_last_n(y, n):
    if len(y) < n: n = len(y)
    t = np.arange(n)
    ys = y[-n:]
    b = np.polyfit(t, ys, 1)[0]
    return b

if __name__ == "__main__":
    print("Analyzing trends over time...")
    cfg = yaml.safe_load(open("src/config.yaml"))
    df = pd.read_parquet("data/interim/clean.parquet")
    y  = pd.read_csv("data/interim/labels.csv", header=None)[0]
    
    # Align lengths to prevent mismatch errors
    n = min(len(df), len(y))
    print(f"Aligning lengths: papers={len(df)}, labels={len(y)}, using={n}")
    df = df.iloc[:n].copy()
    y = y.iloc[:n]
    
    df = df.assign(cluster=y.values)
    df = df.dropna(subset=["date"])
    df["period"] = df["date"].dt.to_period(cfg["time_granularity"]).dt.to_timestamp()

    # counts and shares
    counts = df.groupby(["period","cluster"]).size().rename("n").reset_index()
    totals = counts.groupby("period")["n"].sum().rename("N")
    counts = counts.join(totals, on="period")
    counts["share"] = counts["n"] / counts["N"]

    # trend metrics
    print("Computing trend slopes...")
    out = []
    for c, g in counts.groupby("cluster"):
        g = g.sort_values("period")
        b = slope_last_n(g["share"].values, cfg["trend_window"])
        out.append({"cluster": int(c), "last_share": g["share"].iloc[-1], "n_total": g["n"].sum(), "slope": b})
    trends = pd.DataFrame(out)
    # classify
    ms = cfg["min_support"]; th = cfg["slope_threshold"]
    def cls(row):
        if row["n_total"] < ms: return "tiny"
        if row["slope"] > th: return "rising"
        if row["slope"] < -th: return "declining"
        return "stable"
    trends["class"] = trends.apply(cls, axis=1)
    counts.to_csv("outputs/tables/monthly_counts.csv", index=False)
    trends.to_csv("outputs/tables/trends.csv", index=False)
    print("Trend analysis complete!")
