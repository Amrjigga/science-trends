# Visualization module for creating interactive trend charts
# Generates time-series plots showing cluster share evolution over time with meaningful topic names

import pandas as pd, plotly.express as px, numpy as np

if __name__ == "__main__":
    print("Creating visualizations...")
    counts = pd.read_csv("outputs/tables/monthly_counts.csv", parse_dates=["period"])
    names  = pd.read_csv("outputs/tables/topic_names.csv")
    trends = pd.read_csv("outputs/tables/trends.csv")
    
    # Joining topic names with counts data
    counts = counts.merge(names, how="left", left_on="cluster", right_on="cluster")
    counts["label"] = counts["name"].fillna(counts["cluster"].astype(str))
    
    # Get top rising topics, like stonks going uppp lol
    top = trends.sort_values("slope", ascending=False).head(8)["cluster"].tolist()
    
    # Create viz with meaningful label
    fig = px.line(counts[counts["cluster"].isin(top)],
                  x="period", y="share", color="label",
                  title="Top Rising Research Topics (Share Over Time)")
    fig.write_html("outputs/figures/top_trends.html", include_plotlyjs="cdn")
    print("Visualization saved to outputs/figures/top_trends.html")
