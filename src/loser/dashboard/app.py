import streamlit as st
from loser.dashboard import pipeline
import numpy as np
import os
import subprocess
import pandas as pd
import time

def run_app():
    st.title("LoSeR Dashboard")

    def choose_folder_macos():
        """
        Opens a dialog to choose a folder on MacOS.
        """
        try:
            script = 'POSIX path of (choose folder with prompt "Select the data folder (contains sequences.fasta and clusters.tsv)")'
            out = subprocess.check_output(["osascript", "-e", script])
            return out.decode("utf-8").strip()
        except Exception:
            return None

    st.sidebar.header("Data Selection")
    if "data_folder" not in st.session_state:
        st.session_state["data_folder"] = ""
    pick = st.sidebar.button("Choose folder (macOS Finder)")
    if pick:
        folder = choose_folder_macos()
        if folder != st.session_state["data_folder"]:
            st.cache_data.clear()
        if folder:
            st.session_state["data_folder"] = folder
            st.sidebar.success(f"Selected folder: {folder}")
            st.rerun()
        else:
            st.sidebar.error("No folder selected.")
    data_folder = st.sidebar.text_input("Or enter data folder path", st.session_state["data_folder"])
    st.session_state["data_folder"] = data_folder

    if not data_folder:
        st.info("Select a folder, or paste a path containing sequences.fasta and clusters.tsv to proceed.")
        st.stop()
    seqpath = os.path.join(data_folder, "sequences.fasta")
    clustpath = os.path.join(data_folder, "clusters.tsv")
    if not os.path.isfile(seqpath):
        st.error(f"sequences.fasta not found in {data_folder}")
        st.stop()
    if not os.path.isfile(clustpath):
        st.error(f"clusters.tsv not found in {data_folder}")
        st.stop()

    @st.cache_data(show_spinner=True)
    def load_data(sequences_path, clusters_path):
        genomes = pipeline.read_fasta(sequences_path)
        genomes = pipeline.determine_clusters(clusters_path, genomes)
        return genomes

    genomes = load_data(seqpath, clustpath)
    st.write(f"Total sequences loaded: {len(genomes)}")

    # Build cluster summary
    cluster_counts = {}
    for sid, g in genomes.items():
        c = g.get("cluster")
        if c:
            cluster_counts.setdefault(c, {"count": 0, "example_ids": []})
            entry = cluster_counts[c]
            entry["count"] += 1
            if len(entry["example_ids"]) < 20:
                entry["example_ids"].append(sid)
    st.write(f"Clusters: {len(cluster_counts)}") # Show the clusters

    # Seed for controlling randomness
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=2**32 - 1,
        value=12345,
        step=1,
        key="seed"
    )
    random_state = np.random.RandomState(seed)
    st.write(f"Using seed: {seed}")

    # Number of cores
    avail_cores = os.cpu_count() or 1
    cores = st.slider(
        "Number of CPU cores to use",
        min_value=1,
        max_value=avail_cores,
        value=min(1, avail_cores),
        step=1,
    )

    # Maximum number of genomes per cluster
    max_cluster_size = max((v["count"] for v in cluster_counts.values()), default=0)
    max_genomes = st.slider(
        "Max genomes per cluster to include",
        min_value=1,
        max_value=max_cluster_size,
        value=min(10, max_cluster_size),
        step=1,
        help="Clusters larger than this will be downsampled",
    )

    # Downsample and calculate distances
    if st.button("Downsample and calculate distances"):
        start_time = time.time()
        D, clusters, id2index, index2id = pipeline.downsample_and_compute_distances(genomes, max_genomes=max_genomes, cores=cores)
        end_time = time.time()
        st.success(f"Distance matrix computed ({end_time-start_time:.2f}s) for {len(index2id)} genomes across {len(set(clusters))} clusters.")
        labels = [
            f"{index2id[i]} [{genomes[index2id[i]]['cluster']}]" for i in range(len(index2id))
        ]
        df = pd.DataFrame(D, index=labels, columns=labels)
        st.dataframe(df)

def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False

if _in_streamlit():
    run_app()

"""
Look at this later

selected_cluster = st.selectbox("Select cluster", sorted(cluster_counts.keys()))
info = cluster_counts[selected_cluster]
st.subheader(f"Cluster: {selected_cluster}")
st.write(f"Sequences in cluster: {info['count']}")

# Show example IDs in a scrollable box
ids_text = "\n".join(info["example_ids"])
st.text_area("Example IDs", ids_text, height=220, key=f"examples_{selected_cluster}", disabled=True)
st.code("\n".join(info["example_ids"]), language="text")

# Show detailed sequence selection
seq_id = st.selectbox("Inspect sequence ID", info["example_ids"])
seq_record = genomes[seq_id]
st.write(f"Length: {len(seq_record['sequence'])}")
if st.checkbox("Show sequence"):
    st.code(seq_record["sequence"][:500] + ("..." if len(seq_record["sequence"]) > 500 else ""), language="text")

# Optional: pairwise similarity (simple Jaccard via sourmash)
if st.button("Compute intra-cluster minhash similarity (subset)"):
    subset_ids = info["example_ids"]
    mhs = [genomes[sid]["minhash"] for sid in subset_ids]
    sims = []
    for i in range(len(mhs)):
        row = []
        for j in range(len(mhs)):
            if i == j:
                row.append(1.0)
            else:
                row.append(mhs[i].similarity(mhs[j]))
        sims.append(row)
    st.write("Similarity matrix (example IDs order):")
    st.dataframe(sims)

# Placeholder for integrating Solution
if st.button("Initialize Solution (demo)"):
    from ..solution import Solution
    sol = Solution()  # adapt to required constructor
    st.success("Solution object created")
"""