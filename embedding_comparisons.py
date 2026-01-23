import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from fashion_clip.fashion_clip import FashionCLIP
from transformers import AutoProcessor, AutoModel

# ------------------------------------------------------------------
# 1. Input descriptions
# ------------------------------------------------------------------
descriptions = [
    "white shirt with red dots",                                   # naive
    "white short sleeved shirt with red polka dots",               # fashion-aware
    "round collared regular fit white cotton shirt with micro red polka print"  # fashionista
]

# ============================================================================
# FASHION-CLIP
# ============================================================================
print("=== Fashion-CLIP ===")
fclip = FashionCLIP('fashion-clip')
fclip_embs = fclip.encode_text(descriptions, batch_size=3)
fclip_embs = fclip_embs / np.linalg.norm(fclip_embs, axis=-1, keepdims=True)

fclip_sim_nm = cosine_similarity([fclip_embs[0]], [fclip_embs[1]])[0][0]
fclip_sim_nf = cosine_similarity([fclip_embs[0]], [fclip_embs[2]])[0][0]
fclip_sim_mf = cosine_similarity([fclip_embs[1]], [fclip_embs[2]])[0][0]
fclip_avg = np.mean([fclip_sim_nm, fclip_sim_nf, fclip_sim_mf])

print(f"naive ↔ medium        : {fclip_sim_nm:.4f}")
print(f"naive ↔ fashionista   : {fclip_sim_nf:.4f}")
print(f"medium ↔ fashionista  : {fclip_sim_mf:.4f}")
print(f"AVERAGE               : {fclip_avg:.4f}")
print()

# ============================================================================
# MARQO-FASHIONSIGLIP (Your working syntax)
# ============================================================================
print("=== Marqo-FashionSigLIP ===")
model = AutoModel.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True)
model.eval()

inputs = processor(text=descriptions, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    marqo_embs = model.get_text_features(inputs["input_ids"], normalize=True).cpu().numpy()

marqo_sim_nm = cosine_similarity([marqo_embs[0]], [marqo_embs[1]])[0][0]
marqo_sim_nf = cosine_similarity([marqo_embs[0]], [marqo_embs[2]])[0][0]
marqo_sim_mf = cosine_similarity([marqo_embs[1]], [marqo_embs[2]])[0][0]
marqo_avg = np.mean([marqo_sim_nm, marqo_sim_nf, marqo_sim_mf])

print(f"naive ↔ medium        : {marqo_sim_nm:.4f}")
print(f"naive ↔ fashionista   : {marqo_sim_nf:.4f}")
print(f"medium ↔ fashionista  : {marqo_sim_mf:.4f}")
print(f"AVERAGE               : {marqo_avg:.4f}")
print()

# ============================================================================
# COMPARISON
# ============================================================================
print("=== COMPARISON ===")
print(f"Fashion-CLIP average:     {fclip_avg:.4f}")
print(f"Marqo-FashionSigLIP avg:  {marqo_avg:.4f}")
print(f"Difference:               {abs(fclip_avg - marqo_avg):.4f}")
better = "Fashion-CLIP" if fclip_avg > marqo_avg else "Marqo-FashionSigLIP"
print(f"WINNER (higher avg better): {better}")

# ============================================================================
# EXAMPLE 2:
# ============================================================================
print("TEST 2: ASYMMETRIC HEMLINE DRESS")
print("="*50)

descriptions_2 = [
    "black dress",                                           # naive
    "black asymmetric hemline dress",                        # fashion-aware
    "jet black matte crepe dress, diagonal asymmetric hemline with cascading layers, modern minimalist cut"  # fashionista
]

# FASHION-CLIP
fclip = FashionCLIP('fashion-clip')
fclip_embs_2 = fclip.encode_text(descriptions_2, batch_size=3)
fclip_embs_2 = fclip_embs_2 / np.linalg.norm(fclip_embs_2, axis=-1, keepdims=True)

fclip_sim_nm_2 = cosine_similarity([fclip_embs_2[0]], [fclip_embs_2[1]])[0][0]
fclip_sim_nf_2 = cosine_similarity([fclip_embs_2[0]], [fclip_embs_2[2]])[0][0]
fclip_sim_mf_2 = cosine_similarity([fclip_embs_2[1]], [fclip_embs_2[2]])[0][0]
fclip_avg_2 = np.mean([fclip_sim_nm_2, fclip_sim_nf_2, fclip_sim_mf_2])

print(f"=== Fashion-CLIP ===")
print(f"naive ↔ medium        : {fclip_sim_nm_2:.4f}")
print(f"naive ↔ fashionista   : {fclip_sim_nf_2:.4f}")
print(f"medium ↔ fashionista  : {fclip_sim_mf_2:.4f}")
print(f"AVERAGE               : {fclip_avg_2:.4f}")
print()

# MARQO-FASHIONSIGLIP
model = AutoModel.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True)
model.eval()

inputs_2 = processor(text=descriptions_2, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    marqo_embs_2 = model.get_text_features(inputs_2["input_ids"], normalize=True).cpu().numpy()

marqo_sim_nm_2 = cosine_similarity([marqo_embs_2[0]], [marqo_embs_2[1]])[0][0]
marqo_sim_nf_2 = cosine_similarity([marqo_embs_2[0]], [marqo_embs_2[2]])[0][0]
marqo_sim_mf_2 = cosine_similarity([marqo_embs_2[1]], [marqo_embs_2[2]])[0][0]
marqo_avg_2 = np.mean([marqo_sim_nm_2, marqo_sim_nf_2, marqo_sim_mf_2])

print(f"=== Marqo-FashionSigLIP ===")
print(f"naive ↔ medium        : {marqo_sim_nm_2:.4f}")
print(f"naive ↔ fashionista   : {marqo_sim_nf_2:.4f}")
print(f"medium ↔ fashionista  : {marqo_sim_mf_2:.4f}")
print(f"AVERAGE               : {marqo_avg_2:.4f}")
print()

import pandas as pd

df = pd.read_csv("/content/women_kurta_suit_sets_tags.csv")
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

import re

def normalize_phrase(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)   # normalize spaces
    return text

normalized_tags = [normalize_phrase(t) for t in raw_tags]

from fashion_clip.fashion_clip import FashionCLIP
import torch
import numpy as np

fclip = FashionCLIP('fashion-clip')
fclip_text_embeddings = fclip.encode_text(normalized_tags, batch_size=32)
fclip_text_embeddings = fclip_text_embeddings / np.linalg.norm(
    fclip_text_embeddings, ord=2, axis=-1, keepdims=True
)

import torch
from transformers import AutoModel, AutoProcessor


model = AutoModel.from_pretrained(
    "Marqo/marqo-fashionSigLIP",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "Marqo/marqo-fashionSigLIP",
    trust_remote_code=True
)

model.eval()

import numpy as np

all_embeddings = []

BATCH_SIZE = 32  # safe on CPU

for i in range(0, len(normalized_tags), BATCH_SIZE):
    batch_texts = normalized_tags[i : i + BATCH_SIZE]

    inputs = processor(
        text=batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        batch_embeddings = model.get_text_features(
            inputs["input_ids"],
            normalize=True
        )

    all_embeddings.append(batch_embeddings.cpu().numpy())

text_embeddings = np.vstack(all_embeddings)

print(f"Fashion-CLIP embeddings shape: {fclip_text_embeddings.shape}")
print(f"Marqo embeddings shape: {text_embeddings.shape}")
print(f"Number of tags: {len(normalized_tags)}")

# ============================================================================
# QUERY TEXT
# ============================================================================
# Enter your test query here
QUERY_TEXT = "red kurta"  

print(f"\n🔍 QUERY: '{QUERY_TEXT}'")
print("="*80)

from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# FASHION-CLIP 2.0 TOP 10 CLOSEST (FIXED)
# ============================================================================
print("\n1. FASHION-CLIP 2.0 RESULTS")
print("-" * 60)

# Encode query (FIX: add batch_size=1)
fclip_query_emb = fclip.encode_text([QUERY_TEXT], batch_size=1)[0]
fclip_query_emb = fclip_query_emb / np.linalg.norm(fclip_query_emb)

# Compute distances
fclip_euclidean_dist = euclidean_distances([fclip_query_emb], fclip_text_embeddings)[0]
fclip_manhattan_dist = manhattan_distances([fclip_query_emb], fclip_text_embeddings)[0]
fclip_cosine_sim = cosine_similarity([fclip_query_emb], fclip_text_embeddings)[0]

# Get top 10 closest indices
top10_indices = np.argsort(fclip_euclidean_dist)[:10]

print("Rank | Euclidean | Manhattan | Cosine Sim | Tag")
print("-" * 60)
for rank, idx in enumerate(top10_indices, 1):
    print(f"{rank:4d} | {fclip_euclidean_dist[idx]:9.4f} | {fclip_manhattan_dist[idx]:9.4f} | "
          f"{fclip_cosine_sim[idx]:9.4f} | {normalized_tags[idx]}")

print(f"\n📊 Fashion-CLIP Summary:")
print(f"  Top 1 cosine similarity: {fclip_cosine_sim[top10_indices[0]]:.4f}")
print(f"  Top 10 average cosine similarity: {np.mean(fclip_cosine_sim[top10_indices]):.4f}")

# ============================================================================
# MARQO-FASHIONSIGLIP TOP 10 CLOSEST
# ============================================================================
print("\n\n2. MARQO-FASHIONSIGLIP RESULTS")
print("-" * 60)

# Encode query with Marqo
inputs = processor(text=[QUERY_TEXT], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    marqo_query_emb = model.get_text_features(inputs["input_ids"], normalize=True).cpu().numpy()[0]

# Compute distances
marqo_euclidean_dist = euclidean_distances([marqo_query_emb], text_embeddings)[0]
marqo_manhattan_dist = manhattan_distances([marqo_query_emb], text_embeddings)[0]
marqo_cosine_sim = cosine_similarity([marqo_query_emb], text_embeddings)[0]

# Get top 10 closest indices
marqo_top10_indices = np.argsort(marqo_euclidean_dist)[:10]

print("Rank | Euclidean | Manhattan | Cosine Sim | Tag")
print("-" * 60)
for rank, idx in enumerate(marqo_top10_indices, 1):
    print(f"{rank:4d} | {marqo_euclidean_dist[idx]:9.4f} | {marqo_manhattan_dist[idx]:9.4f} | "
          f"{marqo_cosine_sim[idx]:9.4f} | {normalized_tags[idx]}")

print(f"\n📊 Marqo-FashionSigLIP Summary:")
print(f"  Top 1 cosine similarity: {marqo_cosine_sim[marqo_top10_indices[0]]:.4f}")
print(f"  Top 10 average cosine similarity: {np.mean(marqo_cosine_sim[marqo_top10_indices]):.4f}")

# ============================================================================
# QUERY 2
# ============================================================================
print("\n" + "="*80)
print("🔍 QUERY 2")
print("="*80)
QUERY_TEXT_2 = "anarkali set"

print(f"Query: '{QUERY_TEXT_2}'")
print("\n1. FASHION-CLIP 2.0 RESULTS")
print("-" * 60)

# Encode query (batch_size=1 fix)
fclip_query_emb_2 = fclip.encode_text([QUERY_TEXT_2], batch_size=1)[0]
fclip_query_emb_2 = fclip_query_emb_2 / np.linalg.norm(fclip_query_emb_2)

# Compute distances
fclip_euclidean_dist_2 = euclidean_distances([fclip_query_emb_2], fclip_text_embeddings)[0]
fclip_manhattan_dist_2 = manhattan_distances([fclip_query_emb_2], fclip_text_embeddings)[0]
fclip_cosine_sim_2 = cosine_similarity([fclip_query_emb_2], fclip_text_embeddings)[0]

# Top 10
top10_indices_2 = np.argsort(fclip_euclidean_dist_2)[:10]

print("Rank | Euclidean | Manhattan | Cosine Sim | Tag")
print("-" * 60)
for rank, idx in enumerate(top10_indices_2, 1):
    print(f"{rank:4d} | {fclip_euclidean_dist_2[idx]:9.4f} | {fclip_manhattan_dist_2[idx]:9.4f} | "
          f"{fclip_cosine_sim_2[idx]:9.4f} | {normalized_tags[idx]}")

print(f"\n📊 Fashion-CLIP Summary:")
print(f"  Top 1 cosine: {fclip_cosine_sim_2[top10_indices_2[0]]:.4f}")
print(f"  Top 10 avg cosine: {np.mean(fclip_cosine_sim_2[top10_indices_2]):.4f}")

print("\n2. MARQO-FASHIONSIGLIP RESULTS")
print("-" * 60)

# Marqo query
inputs_2 = processor(text=[QUERY_TEXT_2], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    marqo_query_emb_2 = model.get_text_features(inputs_2["input_ids"], normalize=True).cpu().numpy()[0]

marqo_euclidean_dist_2 = euclidean_distances([marqo_query_emb_2], text_embeddings)[0]
marqo_manhattan_dist_2 = manhattan_distances([marqo_query_emb_2], text_embeddings)[0]
marqo_cosine_sim_2 = cosine_similarity([marqo_query_emb_2], text_embeddings)[0]

marqo_top10_indices_2 = np.argsort(marqo_euclidean_dist_2)[:10]

print("Rank | Euclidean | Manhattan | Cosine Sim | Tag")
print("-" * 60)
for rank, idx in enumerate(marqo_top10_indices_2, 1):
    print(f"{rank:4d} | {marqo_euclidean_dist_2[idx]:9.4f} | {marqo_manhattan_dist_2[idx]:9.4f} | "
          f"{marqo_cosine_sim_2[idx]:9.4f} | {normalized_tags[idx]}")

print(f"\n📊 Marqo Summary:")
print(f"  Top 1 cosine: {marqo_cosine_sim_2[marqo_top10_indices_2[0]]:.4f}")
print(f"  Top 10 avg cosine: {np.mean(marqo_cosine_sim_2[marqo_top10_indices_2]):.4f}")

# ============================================================================
# QUERY 3:
# ============================================================================
print("\n" + "="*80)
print("🔍 QUERY 3")
print("="*80)
QUERY_TEXT_3 = "Summer Wear"

print(f"Query: '{QUERY_TEXT_3}'")
print("\n1. FASHION-CLIP 2.0 RESULTS")
print("-" * 60)

# Encode query
fclip_query_emb_3 = fclip.encode_text([QUERY_TEXT_3], batch_size=1)[0]
fclip_query_emb_3 = fclip_query_emb_3 / np.linalg.norm(fclip_query_emb_3)

# Distances
fclip_euclidean_dist_3 = euclidean_distances([fclip_query_emb_3], fclip_text_embeddings)[0]
fclip_manhattan_dist_3 = manhattan_distances([fclip_query_emb_3], fclip_text_embeddings)[0]
fclip_cosine_sim_3 = cosine_similarity([fclip_query_emb_3], fclip_text_embeddings)[0]

top10_indices_3 = np.argsort(fclip_euclidean_dist_3)[:10]

print("Rank | Euclidean | Manhattan | Cosine Sim | Tag")
print("-" * 60)
for rank, idx in enumerate(top10_indices_3, 1):
    print(f"{rank:4d} | {fclip_euclidean_dist_3[idx]:9.4f} | {fclip_manhattan_dist_3[idx]:9.4f} | "
          f"{fclip_cosine_sim_3[idx]:9.4f} | {normalized_tags[idx]}")

print(f"\n📊 Fashion-CLIP Summary:")
print(f"  Top 1 cosine: {fclip_cosine_sim_3[top10_indices_3[0]]:.4f}")
print(f"  Top 10 avg cosine: {np.mean(fclip_cosine_sim_3[top10_indices_3]):.4f}")

print("\n2. MARQO-FASHIONSIGLIP RESULTS")
print("-" * 60)

# Marqo query
inputs_3 = processor(text=[QUERY_TEXT_3], padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    marqo_query_emb_3 = model.get_text_features(inputs_3["input_ids"], normalize=True).cpu().numpy()[0]

marqo_euclidean_dist_3 = euclidean_distances([marqo_query_emb_3], text_embeddings)[0]
marqo_manhattan_dist_3 = manhattan_distances([marqo_query_emb_3], text_embeddings)[0]
marqo_cosine_sim_3 = cosine_similarity([marqo_query_emb_3], text_embeddings)[0]

marqo_top10_indices_3 = np.argsort(marqo_euclidean_dist_3)[:10]

print("Rank | Euclidean | Manhattan | Cosine Sim | Tag")
print("-" * 60)
for rank, idx in enumerate(marqo_top10_indices_3, 1):
    print(f"{rank:4d} | {marqo_euclidean_dist_3[idx]:9.4f} | {marqo_manhattan_dist_3[idx]:9.4f} | "
          f"{marqo_cosine_sim_3[idx]:9.4f} | {normalized_tags[idx]}")

print(f"\n📊 Marqo Summary:")
print(f"  Top 1 cosine: {marqo_cosine_sim_3[marqo_top10_indices_3[0]]:.4f}")
print(f"  Top 10 avg cosine: {np.mean(marqo_cosine_sim_3[marqo_top10_indices_3]):.4f}")



import hdbscan
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ============================================================================
# HDBSCAN + DBCV (SIMPLIFIED - FIXED)
# ============================================================================
print("\n" + "="*80)
print("🎯 HDBSCAN + DBCV (Density-Based Clustering)")
print("="*80)


hdbscan_fclip = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric='euclidean',
    cluster_selection_method='eom',
    gen_min_span_tree=True # Explicitly set
)
fclip_labels = hdbscan_fclip.fit(fclip_text_embeddings)
fclip_dbcv = hdbscan_fclip.relative_validity_

hdbscan_maro = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric='euclidean',
    cluster_selection_method='eom',
    gen_min_span_tree=True # Explicitly set
)
marqo_labels = hdbscan_maro.fit(text_embeddings)
marqo_dbcv = hdbscan_maro.relative_validity_

print(f"\nFashion-CLIP 2.0:")
print(f"  DBCV Score: {fclip_dbcv:.4f}")
print(f"  Clusters: {len(set(fclip_labels.labels_)) - (1 if -1 in fclip_labels.labels_ else 0)}")
print(f"  Noise: {list(fclip_labels.labels_).count(-1)}")

print(f"\nMarqo-FashionSigLIP:")
print(f"  DBCV Score: {marqo_dbcv:.4f}")
print(f"  Clusters: {len(set(marqo_labels.labels_)) - (1 if -1 in marqo_labels.labels_ else 0)}")
print(f"  Noise: {list(marqo_labels.labels_).count(-1)}")

winner = "FCLIP 🏆" if fclip_dbcv > marqo_dbcv else "Marqo 🏆"
print(f"\n🏆 DBCV WINNER: {winner}")

print(f"\n📊 Interpretation:")
print(f"  > 0.70: Excellent")
print(f"  0.40-0.70: Good")
print(f"  0.10-0.40: Fair")

# ============================================================================
# VISUALIZATION (Optional - Fixed)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
fclip_2d = tsne.fit_transform(fclip_text_embeddings[:2000])  # Subsample for speed
marqo_2d = tsne.fit_transform(text_embeddings[:2000])

ax1.scatter(fclip_2d[:, 0], fclip_2d[:, 1], c=fclip_labels.labels_[:2000],
            cmap='tab10', alpha=0.7, s=20)
ax1.set_title(f'Fashion-CLIP HDBSCAN\nDBCV={fclip_dbcv:.3f}', fontweight='bold')

ax2.scatter(marqo_2d[:, 0], marqo_2d[:, 1], c=marqo_labels.labels_[:2000],
            cmap='tab10', alpha=0.7, s=20)
ax2.set_title(f'Marqo HDBSCAN\nDBCV={marqo_dbcv:.3f}', fontweight='bold')

plt.tight_layout()
plt.savefig('hdbscan_fixed.png', dpi=300, bbox_inches='tight')
print("✅ Plot saved: hdbscan_fixed.png")
plt.show()

