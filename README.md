# 📝 Anchor Tag Registry for Fashion Attributes

A system for standardizing noisy fashion tags into an interpretable anchor/canonical registry using semantic embeddings, density-based clustering, and controlled registry logic.
It converts thousands of inconsistent raw tags into a stable set of meaningful canonical tags, while preserving semantic nuance and avoiding over-merging.

## 🔁 Pipeline Flow

## Part 1 – Embedding Layer

We convert free-text tags into **semantic vectors** using a fashion-specific multimodal model, so that similar tags live close together in embedding space.


## Part 2 – Density-Based Clustering (HDBSCAN)

### What happens here?

We cluster raw tags into groups of **semantically dense** tags, so that each cluster represents a coherent fashion concept.

### Why HDBSCAN?

- Automatically finds a suitable number of clusters    
- Works well with high-dimensional embeddings  
- Is **density-aware**

### Noise handling

- Noise points (label `-1`) are converted into **singleton clusters**  
- Every tag belongs to some cluster in the final mapping  
- No tag is discarded from the vocabulary


## Part 3 – Canonical Registry Formation  
*(Canonical → Final Canonical)*

Canonical tags from Part 2 are registered into a **controlled vocabulary** using a hybrid strategy based on:

- **Semantic similarity** (embedding cosine similarity)  
- **Lexical similarity** (e.g., RapidFuzz token-based matching)

Additional rule:

- Large, high-support canonical clusters are **preserved as independent final canonicals**

This design **reduces greedy chaining** while still **merging genuinely equivalent concepts**, giving a compact and interpretable set of final canonical tags.

## 📦 Output
https://drive.google.com/drive/folders/1quBZ0NfxY52pdXmJY-274GrDhPU-gf_V?usp=sharing


## 📚 Resources

###  YouTube

- [Clustering with DBSCAN, Clearly Explained!!!](https://www.youtube.com/watch?v=RDZUdRSDOok)  
  Intuitive, visual explanation of how DBSCAN works, including core points, density, and handling outliers.[file:1]

- [Agglomerative Clustering: How It Works](https://www.youtube.com/watch?v=XJ3194AmH40)  
  Clear walkthrough of bottom-up hierarchical clustering and dendrograms, with complexity discussion.[file:2]

###  Articles & Guides

- [Davies–Bouldin Index (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/davies-bouldin-index/)  
  Explains the Davies–Bouldin index for evaluating cluster compactness and separation with examples.[web:204]

- [Calinski–Harabasz Index (GeeksforGeeks)](https://www.geeksforgeeks.org/machine-learning/calinski-harabasz-index-cluster-validity-indices-set-3/)  
  Describes the Calinski–Harabasz score and how it measures between- vs within-cluster dispersion.[web:222]

- [Understanding Silhouette Score in Clustering (Medium)](https://farshadabdulazeez.medium.com/understanding-silhouette-score-in-clustering-8aedc06ce9c4)  
  Intuitive explanation of the silhouette score, its formula, and how to interpret values in practice.[web:202]

