"""
Consensus Sampling Pipeline
============================
Munavar's "best output through convergence" idea — lean version.

How it works:
1. Run the same prompt N times in parallel against a local Ollama model
2. Embed all outputs using a sentence-transformer model
3. Cluster outputs by semantic similarity (Average or DBSCAN)
4. Project embeddings to 2D coordinates using PCA for visual tracking
5. Pick the centroid of the densest cluster as the "consensus winner"
6. Run a refinement loop ("make it better") on that winner
7. Export run metadata to a JSON packet compatible with the visual cockpit
8. Return the final optimised output

Requirements:
    pip install ollama sentence-transformers scikit-learn numpy

Ollama must be running locally:
    ollama serve
    ollama pull mistral   (or any model you like)
"""

import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys
import json
import os
import re

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "mistral"          # change to any ollama model
EMBED_MODEL        = "all-MiniLM-L6-v2" # fast, light, runs on CPU
N_GENERATIONS      = 20                 # number of parallel samples
SIMILARITY_THRESH  = 0.80               # cosine sim threshold for "same cluster"
REFINE_ROUNDS      = 2                  # how many "make it better" passes
TEMPERATURE        = 0.8                # sampling temperature (diversity)
# ─────────────────────────────────────────────────────────────────────────────


def generate_outputs(prompt: str, model: str, n: int, temperature: float = TEMPERATURE, num_workers: int = 4) -> list[str]:
    """Generate N independent outputs for the same prompt in parallel."""
    outputs = [None] * n
    print(f"\n[1/4] Generating {n} outputs from '{model}' in parallel with {num_workers} workers...")
    
    def generate_single(index):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature}
            )
            content = response["message"]["content"].strip()
            return index, content
        except Exception as e:
            print(f"\n      [Warning] Generation failed for Sample {index+1}: {e}")
            # Fallback placeholder text to avoid crashing thread pool
            return index, f"[Generation Error] Failed to generate sample {index+1} from model '{model}'."

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(generate_single, i): i for i in range(n)}
        completed = 0
        for future in as_completed(futures):
            idx, content = future.result()
            outputs[idx] = content
            completed += 1
            print(f"      Sample {completed}/{n} completed", end="\r")
            
    print(f"      Done - {n} outputs collected.          ")
    return outputs


def embed_outputs(outputs: list[str], embed_model: str) -> np.ndarray:
    """Convert text outputs to semantic embedding vectors."""
    print(f"\n[2/4] Embedding outputs with '{embed_model}'...")
    model = SentenceTransformer(embed_model)
    embeddings = model.encode(outputs, show_progress_bar=False)
    print(f"      Embedding shape: {embeddings.shape}")
    return embeddings


def find_consensus(outputs: list[str], embeddings: np.ndarray, clustering_method: str = "average") -> tuple[int, np.ndarray, np.ndarray]:
    """
    Find the output that represents the consensus centroid.
    Method 'average': average cosine similarity to all others.
    Method 'dbscan': run DBSCAN first to isolate the dense core cluster, then find the centroid of that cluster.
    Returns (best_idx, avg_similarities, sim_matrix)
    """
    print(f"\n[3/4] Computing similarity matrix and finding consensus ({clustering_method} clustering)...")
    sim_matrix = cosine_similarity(embeddings)       # N x N matrix
    N = len(outputs)
    
    # Calculate global average similarities first for individual scoring
    avg_similarities = sim_matrix.mean(axis=1)
    
    if clustering_method == "dbscan":
        # Convert similarity to distance matrix
        distance_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
        
        # Run DBSCAN (eps=0.25 corresponds to a cosine similarity threshold of 0.75)
        # min_samples is dynamic based on sample size
        db = DBSCAN(eps=0.25, min_samples=max(2, int(N * 0.15)), metric="precomputed")
        labels = db.fit_predict(distance_matrix)
        
        # Find unique clusters excluding noise (-1)
        unique_labels = set(labels) - {-1}
        
        if len(unique_labels) > 0:
            # Find the largest cluster
            largest_label = max(unique_labels, key=lambda l: np.sum(labels == l))
            cluster_indices = np.where(labels == largest_label)[0]
            print(f"      DBSCAN isolated largest core cluster with {len(cluster_indices)}/{N} members.")
            
            # Find the centroid of the largest cluster
            # This is the member of the cluster that has the highest average similarity to other cluster members
            cluster_sims = sim_matrix[cluster_indices][:, cluster_indices]
            cluster_avg_sims = cluster_sims.mean(axis=1)
            best_cluster_idx = np.argmax(cluster_avg_sims)
            best_idx = int(cluster_indices[best_cluster_idx])
        else:
            print("      DBSCAN failed to identify distinct clusters (all items classified as noise).")
            print("      Falling back to global average centroid selection...")
            best_idx = int(np.argmax(avg_similarities))
    else:
        best_idx = int(np.argmax(avg_similarities))
        
    best_score = float(avg_similarities[best_idx])

    # Show cluster stats
    print(f"      Similarity scores (avg per output):")
    for i, score in enumerate(avg_similarities):
        marker = " <- WINNER" if i == best_idx else ""
        print(f"        Output {i+1:02d}: {score:.4f}{marker}")

    return best_idx, avg_similarities, sim_matrix


def refine_output(winner: str, prompt: str, model: str, rounds: int) -> tuple[list[str], str]:
    """Run 'make it better' refinement loop on the consensus winner. Returns (refinement_rounds_texts, final_output)"""
    print(f"\n[4/4] Running {rounds} refinement round(s)...")
    refinements = []
    current = winner
    for r in range(rounds):
        print(f"      Refinement round {r+1}/{rounds}...")
        refine_prompt = (
            f"Original task: {prompt}\n\n"
            f"Current output:\n{current}\n\n"
            "Make it better. Improve clarity, depth, and quality. "
            "Return only the improved output."
        )
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": refine_prompt}],
                options={"temperature": 0.4}   # lower temp for refinement = more focused
            )
            current = response["message"]["content"].strip()
            refinements.append(current)
        except Exception as e:
            print(f"      [Warning] Refinement round {r+1} failed: {e}")
            refinements.append(f"[Refinement Error] Failed to execute round {r+1} refinement.")
            
    return refinements, current


def project_embeddings_2d(embeddings: np.ndarray) -> np.ndarray:
    """Project high-dim embeddings into normalized 2D coordinates for visualizer chart scaling."""
    if len(embeddings) < 2:
        return np.zeros((len(embeddings), 2))
        
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings) # shape: N x 2
    
    # Scale coordinates to fit beautifully in SVG bounds between [-80, 80]
    min_vals = coords_2d.min(axis=0)
    max_vals = coords_2d.max(axis=0)
    ranges = max_vals - min_vals
    
    # Avoid divide-by-zero if all coordinates are identical
    ranges[ranges == 0.0] = 1.0
    
    normalized = -80.0 + ((coords_2d - min_vals) / ranges) * 160.0
    return normalized


def export_run_data(filepath: str, prompt: str, model: str, outputs: list[str],
                    embeddings: np.ndarray, sim_matrix: np.ndarray,
                    avg_similarities: np.ndarray, winner_idx: int,
                    refinements: list[str], final_output: str):
    """Export structured run data to a JSON file for visualizer compatibility."""
    print(f"\n[Export] Saving run metadata to '{filepath}'...")
    
    # Project embeddings to 2D
    coords_2d = project_embeddings_2d(embeddings)
    
    samples_list = []
    for i in range(len(outputs)):
        samples_list.append({
            "id": i + 1,
            "text": outputs[i],
            "avg_similarity": float(avg_similarities[i]),
            "is_winner": (i == winner_idx),
            "coordinates": [float(coords_2d[i][0]), float(coords_2d[i][1])]
        })
        
    run_data = {
        "prompt": prompt,
        "model": model,
        "n_samples": len(outputs),
        "refine_rounds": len(refinements),
        "winner_id": winner_idx,
        "raw_winner": outputs[winner_idx],
        "generations": outputs,
        "refinements": refinements,
        "final_output": final_output,
        "samples": samples_list,
        "similarity_matrix": sim_matrix.tolist()
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2)
    print(f"      Run data exported successfully!")


def sanitize_filename(prompt: str) -> str:
    """Convert prompt to a safe, clean filename for saving run data."""
    # Strip whitespace
    s = prompt.strip()
    # Remove any non-alphanumeric characters, except spaces and hyphens
    s = re.sub(r'[^\w\s-]', '', s)
    # Replace spaces and hyphens with underscores
    s = re.sub(r'[\s-]+', '_', s)
    # Cap length at 60 characters to prevent overly long filenames
    s = s[:60].strip('_')
    # If empty, use fallback
    if not s:
        s = "query_run"
    return f"{s}.json"


def run_pipeline(prompt: str, model: str = DEFAULT_MODEL,
                 n: int = N_GENERATIONS, refine: int = REFINE_ROUNDS,
                 workers: int = 4, clustering: str = "average",
                 export_path: str = None) -> str:
    """Full consensus pipeline. Returns the final optimised output."""
    print("=" * 60)
    print("  CONSENSUS SAMPLING PIPELINE")
    print("=" * 60)
    print(f"  Prompt     : {prompt[:80]}{'...' if len(prompt)>80 else ''}")
    print(f"  Model      : {model}")
    print(f"  Samples    : {n}")
    print(f"  Workers    : {workers}")
    print(f"  Clustering : {clustering}")
    print(f"  Refine     : {refine} round(s)")
    print("=" * 60)

    outputs    = generate_outputs(prompt, model, n, num_workers=workers)
    embeddings = embed_outputs(outputs, EMBED_MODEL)
    
    winner_idx, avg_similarities, sim_matrix = find_consensus(outputs, embeddings, clustering_method=clustering)
    winner = outputs[winner_idx]
    score = avg_similarities[winner_idx]

    print(f"\n  Consensus winner similarity score: {score:.4f}")
    print(f"\n  RAW WINNER:\n  {'-'*40}")
    print(winner[:400] + ("..." if len(winner) > 400 else ""))

    refinements = []
    if refine > 0:
        refinements, final = refine_output(winner, prompt, model, refine)
    else:
        final = winner

    print(f"\n{'='*60}")
    print("  FINAL OPTIMISED OUTPUT")
    print(f"{'='*60}")
    print(final)
    print(f"{'='*60}\n")

    # Always save run data to a folder called 'queries' in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    queries_dir = os.path.join(project_root, "queries")
    if not os.path.exists(queries_dir):
        try:
            os.makedirs(queries_dir)
            print(f"      Created 'queries/' directory in project root.")
        except Exception as e:
            print(f"[Warning] Failed to create 'queries' directory: {e}")
            
    if os.path.exists(queries_dir):
        filename = sanitize_filename(prompt)
        auto_export_path = os.path.join(queries_dir, filename)
        try:
            export_run_data(auto_export_path, prompt, model, outputs, embeddings,
                            sim_matrix, avg_similarities, winner_idx,
                            refinements, final)
        except Exception as e:
            print(f"[Warning] Failed to auto-save run data to '{auto_export_path}': {e}")

    if export_path:
        export_run_data(export_path, prompt, model, outputs, embeddings,
                        sim_matrix, avg_similarities, winner_idx,
                        refinements, final)

    return final


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consensus Sampling Pipeline")
    parser.add_argument("prompt",  type=str, help="The task or question to optimise")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument("--n",     type=int, default=N_GENERATIONS,  help="Number of samples")
    parser.add_argument("--refine",type=int, default=REFINE_ROUNDS,  help="Refinement rounds")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for Ollama generation")
    parser.add_argument("--clustering", type=str, choices=["average", "dbscan"], default="average", help="Clustering method for consensus")
    parser.add_argument("--export", type=str, default=None, help="Filepath to export run metadata JSON for visualizer")
    args = parser.parse_args()

    result = run_pipeline(
        prompt=args.prompt,
        model=args.model,
        n=args.n,
        refine=args.refine,
        workers=args.workers,
        clustering=args.clustering,
        export_path=args.export
    )
    sys.exit(0)
