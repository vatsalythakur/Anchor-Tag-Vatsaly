import pandas as pd

df = pd.read_csv("women_kurta_suit_sets_tags.csv")
df.head()

df.shape
len(df['normalised_name'].unique())

raw_tags = (
    df["normalised_name"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)
len(raw_tags), raw_tags[:10]

import re

def normalize_phrase(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)   # normalize spaces
    return text

normalized_tags = [normalize_phrase(t) for t in raw_tags]

pd.Series(normalized_tags).sample(10)

import hdbscan
from sklearn.cluster import AgglomerativeClustering
from rapidfuzz import fuzz

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/data/similars/ajio/fashion-clip")
print(tok.vocab_size)

import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# ---- Load model (once) ----
processor = AutoProcessor.from_pretrained("/data/similars/ajio/fashion-clip")
model = AutoModelForZeroShotImageClassification.from_pretrained(
    "/data/similars/ajio/fashion-clip"
)

model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ---- Text embedding function (FashionCLIP equivalent) ----
def encode_text_fashionclip(texts, batch_size=32):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # FashionCLIP exposes text features like CLIP
            text_features = model.get_text_features(**inputs)

            # Normalize (important for cosine similarity)
            text_features = torch.nn.functional.normalize(
                text_features, dim=1
            )

            all_embeddings.append(text_features.cpu().numpy())

    return np.vstack(all_embeddings)

# ---- EXACT SAME variable name as before ----
text_embeddings = encode_text_fashionclip(
    normalized_tags,
    batch_size=32
)

# (Optional safety – already normalized above)
text_embeddings = text_embeddings / np.linalg.norm(
    text_embeddings,
    axis=1,
    keepdims=True
)

import hdbscan
import numpy as np

clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        gen_min_span_tree=True
    )

cluster_labels = clusterer.fit_predict(text_embeddings)

n_clusters_hdbscan = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = np.sum(cluster_labels == -1)

print(f"🔹 HDBSCAN clusters (excluding noise): {n_clusters_hdbscan}")
print(f"🔹 Noise points: {n_noise}")

# Convert noise points (-1) into unique cluster IDs
if n_noise > 0:
    max_cluster_id = cluster_labels[cluster_labels != -1].max() if n_clusters_hdbscan > 0 else -1
    noise_indices = np.where(cluster_labels == -1)[0]

    for i, idx in enumerate(noise_indices):
        cluster_labels[idx] = max_cluster_id + 1 + i

unique_clusters = np.unique(cluster_labels)

print(f"✅ Final total clusters (including singletons): {len(unique_clusters)}")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_part_a(embeddings, labels):
    unique_clusters = np.unique(labels)

    intra_sims = []
    centroids = []

    # ---- Intra-cluster similarity ----
    for cid in unique_clusters:
        mask = labels == cid
        cluster_emb = embeddings[mask]

        if len(cluster_emb) > 1:
            sim_matrix = cosine_similarity(cluster_emb)
            avg_sim = (sim_matrix.sum() - len(cluster_emb)) / (
                len(cluster_emb) * (len(cluster_emb) - 1)
            )
            intra_sims.append(avg_sim)

        # centroid for inter-cluster
        centroids.append(cluster_emb.mean(axis=0))

    intra_avg = np.mean(intra_sims)

    # ---- Inter-cluster similarity (centroids) ----
    centroid_matrix = cosine_similarity(np.vstack(centroids))
    np.fill_diagonal(centroid_matrix, np.nan)
    inter_avg = np.nanmean(centroid_matrix)

    compactness = (intra_avg - inter_avg) / max(intra_avg + inter_avg, 1e-6)

    return {
        "intra_cluster_similarity": intra_avg,
        "inter_cluster_similarity": inter_avg,
        "compactness_score": compactness
    }

part_a_metrics = evaluate_part_a(
    embeddings=text_embeddings,
    labels=cluster_labels
)

print("🔹 Part A Metrics")
for k, v in part_a_metrics.items():
    print(f"{k}: {v:.4f}")

# Build the clusters DataFrame (if not already done)
clusters_df = pd.DataFrame({
    "tag": normalized_tags,
    "cluster": cluster_labels
})

# Add cluster size for sorting
cluster_sizes = clusters_df.groupby("cluster").size().reset_index(name="cluster_size")
clusters_df = clusters_df.merge(cluster_sizes, on="cluster")

# Sort by cluster_id and tag for readability
clusters_df = clusters_df.sort_values(["cluster", "tag"])

# Export to CSV (downloadable from Colab)
clusters_df.to_csv("subset_clusters_formed_HDBSCAN_modify.csv", index=False)

print("✅ Saved 'clusters_complete_test_overview.csv' with all clusters!")
print(f"Total clusters: {clusters_df['cluster'].nunique()}")
print(f"Total tags: {len(clusters_df)}")


from rapidfuzz import fuzz
def pick_fuzzy_canonical(phrases):
    if len(phrases) == 1:
        return phrases[0]

    scores = {}

    for p in phrases:
        total = 0
        for q in phrases:
            if p != q:
                total += fuzz.token_set_ratio(p, q)
        scores[p] = total / (len(phrases) - 1)

    # phrase with highest average similarity
    return max(scores, key=scores.get)

cluster_to_canonical = {}

for cid, group in clusters_df.groupby("cluster"):
    phrases = group["tag"].tolist()
    canonical = pick_fuzzy_canonical(phrases)
    cluster_to_canonical[cid] = canonical

# Attach fuzzy canonical tag
clusters_df["canonical_tag"] = clusters_df["cluster"].map(cluster_to_canonical)

# Sort for human readability
inspection_df = clusters_df[
    ["canonical_tag", "cluster", "cluster_size", "tag"]
].sort_values(
    by=["canonical_tag", "cluster", "tag"]
)

inspection_df.to_csv(
    "fuzzy_canonical_inspection_HDBSCAN_modify.csv",
    index=False
)

print("✅ Saved fuzzy_canonical_inspection.csv")
print(f"Total clusters: {inspection_df['cluster'].nunique()}")
print(f"Total canonical tags: {inspection_df['canonical_tag'].nunique()}")

canonical_names = list(cluster_to_canonical.values())
canonical_names = list(dict.fromkeys(canonical_names))  # unique, preserve order

len(canonical_names)

# ---- Encode canonical cluster names (EXACTLY like before) ----
canonical_embeddings = encode_text_fashionclip(
    canonical_names,
    batch_size=32
)

# (Optional safety – already normalized inside encoder)
canonical_embeddings = canonical_embeddings / np.linalg.norm(
    canonical_embeddings,
    axis=1,
    keepdims=True
)

from collections import Counter

cluster_sizes = Counter(clusters_df["cluster"])

canonical_cluster_sizes = {
    cluster_to_canonical[cid]: cluster_sizes[cid]
    for cid in cluster_to_canonical
}

sorted(canonical_cluster_sizes.items(), key=lambda x: -x[1])[:10]

import numpy as np
from rapidfuzz import fuzz

STRONG_EMB_THRESHOLD = 0.95
WEAK_EMB_THRESHOLD = 0.85
FUZZY_THRESHOLD = 80
ANCHOR_CLUSTER_SIZE = 4

vocab_names = []
vocab_embeddings = []
final_canonical_map = {}

# -------------------------------
# PHASE 1: Anchor strong canonicals
# -------------------------------
for name, emb in zip(canonical_names, canonical_embeddings):
    if canonical_cluster_sizes.get(name, 0) >= ANCHOR_CLUSTER_SIZE:
        vocab_names.append(name)
        vocab_embeddings.append(emb)
        final_canonical_map[name] = name

# -------------------------------
# PHASE 2: Greedy registry for rest
# -------------------------------
for name, emb in zip(canonical_names, canonical_embeddings):

    # already anchored → skip
    if name in final_canonical_map:
        continue

    assigned = False

    if vocab_embeddings:
        sims = np.dot(vocab_embeddings, emb)
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]

        fuzzy_sim = fuzz.token_set_ratio(
            name,
            vocab_names[best_idx]
        )

        if best_sim >= STRONG_EMB_THRESHOLD:
            final_canonical_map[name] = vocab_names[best_idx]
            assigned = True

        elif best_sim >= WEAK_EMB_THRESHOLD and fuzzy_sim >= FUZZY_THRESHOLD:
            final_canonical_map[name] = vocab_names[best_idx]
            assigned = True

    if not assigned:
        vocab_names.append(name)
        vocab_embeddings.append(emb)
        final_canonical_map[name] = name


cluster_to_final_canonical = {
    cid: final_canonical_map[canon]
    for cid, canon in cluster_to_canonical.items()
}

clusters_df["final_canonical_tag"] = clusters_df["cluster"].map(cluster_to_final_canonical)

final_df = clusters_df[
    ["final_canonical_tag", "canonical_tag", "cluster", "tag"]
].sort_values(
    ["final_canonical_tag", "cluster", "tag"]
)

final_df.to_csv(
    "final_canonical_registry__HDBSCAN_Anchor_thres_4_emb_thres_0.95_fuzzy_85.csv",
    index=False
)

print("✅ Saved final_canonical_registry.csv")
print("Total final canonical tags:", final_df["final_canonical_tag"].nunique())


import numpy as np

# Assign numeric labels to final canonicals
final_canonicals = list(dict.fromkeys(final_canonical_map.values()))
final_canonical_to_id = {
    name: idx for idx, name in enumerate(final_canonicals)
}

part_b_labels = np.array([
    final_canonical_to_id[final_canonical_map[name]]
    for name in canonical_names
])

import numpy as np

def l2_normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def fast_intra_similarity(SIM, labels):
    intra_vals = []

    for g in np.unique(labels):
        if g == -1:
            continue

        idx = np.where(labels == g)[0]
        if len(idx) < 2:
            continue

        sub = SIM[np.ix_(idx, idx)]
        n = len(idx)

        # remove diagonal (self similarity = 1)
        avg = (sub.sum() - n) / (n * (n - 1))
        intra_vals.append(avg)

    return np.mean(intra_vals) if intra_vals else 0.0

def compute_centroids(E, labels):
    centroids = {}
    for g in np.unique(labels):
        if g == -1:
            continue
        centroids[g] = E[labels == g].mean(axis=0)
    
    # normalize centroids
    for g in centroids:
        centroids[g] /= np.linalg.norm(centroids[g])
    return centroids

def fast_inter_similarity(centroids):
    keys = list(centroids.keys())
    C = np.vstack([centroids[k] for k in keys])

    SIM_C = C @ C.T
    n = len(keys)

    # exclude diagonal
    return (SIM_C.sum() - n) / (n * (n - 1))

def evaluate_part_b_fast(embeddings, labels):
    E = l2_normalize(embeddings.astype(np.float64))
    SIM = E @ E.T

    intra = fast_intra_similarity(SIM, labels)
    centroids = compute_centroids(E, labels)
    inter = fast_inter_similarity(centroids)

    compactness = (intra - inter) / max(intra + inter, 1e-6)

    return {
        "intra_cluster_similarity": intra,
        "inter_cluster_dissimilarity": inter,
        "compactness_score": compactness
    }

results = evaluate_part_b_fast(
    canonical_embeddings,
    part_b_labels
)

print(f"Intra-cluster similarity : {results['intra_cluster_similarity']:.4f}")
print(f"Inter-cluster similarity : {results['inter_cluster_dissimilarity']:.4f}")
print(f"Compactness score        : {results['compactness_score']:.4f}")

import pandas as pd

final_df = pd.read_csv(
    "final_canonical_registry__HDBSCAN_Anchor_thres_4_emb_thres_0.95_fuzzy_85.csv"
)

# Count raw tags per final canonical
cluster_sizes = (
    final_df
    .groupby("final_canonical_tag")
    .size()
    .reset_index(name="cluster_size")
)

# Sort largest → smallest
cluster_sizes = cluster_sizes.sort_values(
    "cluster_size", ascending=False
).reset_index(drop=True)

# Cumulative raw tag count
cluster_sizes["cumulative_raw_tags"] = cluster_sizes["cluster_size"].cumsum()

# Cumulative percentage
total_tags = cluster_sizes["cluster_size"].sum()
cluster_sizes["cumulative_pct"] = (
    cluster_sizes["cumulative_raw_tags"] / total_tags
)
