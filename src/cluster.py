# Clustering module for grouping similar papers
# Uses MiniBatchKMeans with subsampled silhouette analysis for scalability on large datasets

import time, yaml, numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import hdbscan

# Fit MiniBatch k-means clustering with specified number of clusters (fast for large data)
def fit_minibatch_kmeans(Z, k):
    mb = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=5, random_state=0)
    return mb.fit_predict(Z)

# Fit HDBSCAN clustering for density based grouping
def fit_hdbscan(Z):
    return hdbscan.HDBSCAN(min_cluster_size=50, prediction_data=True).fit_predict(Z)

if __name__ == "__main__":
    print("Starting clustering...")
    cfg = yaml.safe_load(open("src/config.yaml"))
    
    # Load embeddings
    Z = np.load("data/interim/embeddings.npy")
    n = len(Z)
    print(f"Loaded {n} paper embeddings")
    
    if cfg["clusterer"] == "kmeans":
        # Using a SUBSAMPLE so it can run faster
        m = min(30000, n)                           
        rng = np.random.RandomState(0)
        idx = rng.choice(n, m, replace=False)
        Zs = Z[idx]
        
        k0, k1 = cfg["k_range"]
        best_k, best_sil = None, -1.0
        
        print(f"Selecting optimal k on subsample m={m} from N={n}")
        for k in range(k0, k1 + 1):
            t0 = time.time()
            mb = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=5, random_state=0)
            ys = mb.fit_predict(Zs)
            sil = silhouette_score(Zs, ys) if len(set(ys)) > 1 else -1.0
            dt = time.time() - t0
            print(f"k={k:>2}  sil={sil: .4f}  time={dt: .1f}s")
            if sil > best_sil:
                best_k, best_sil = k, sil
        
        print(f"Best k={best_k} with silhouette={best_sil: .4f}")
        
        # Final fit on FULL data with MiniBatchKMeans at best_k
        print(f"Final fit on full data N={n} with k={best_k}")
        final = MiniBatchKMeans(n_clusters=best_k, batch_size=4096, n_init=10, random_state=0)
        y = final.fit_predict(Z)
        
        pd.Series(y).to_csv("data/interim/labels.csv", index=False)
    else:
        print("Using HDBSCAN clustering...")
        y = fit_hdbscan(Z)
        pd.Series(y).to_csv("data/interim/labels.csv", index=False)
    
    print("Clustering complete!")
