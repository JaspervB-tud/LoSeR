from __future__ import annotations
from typing import List, Tuple, Literal
import numpy as np
from Bio import SeqIO
import sourmash
from scipy.spatial.distance import squareform
from multiprocessing import Pool, cpu_count

from ..solution import Solution

SelectionMethod = Literal["random", "centroid"]

def read_fasta(filepath: str, min_length: int = 0) -> dict[str], list[str]:
    """
    Reads a multi-FASTA file and returns a dictionary mapping sequence IDs to sequences.

    Parameters:
    -----------
    filepath: str
        Path to the multi-FASTA file.
    min_length: int
        Minimum length of sequences to include. Sequences shorter than this length will be skipped.
        Default is 0 (no minimum length).

    Returns:
    --------
    genomes: dict [str] -> (SeqRecord, str, sourmash.MinHash)
        Dictionary mapping sequence IDs in the FASTA file to their corresponding sequences.
    genome_ids: list[str]
        List of sequence IDs.
    """
    genomes = {}
    for record in SeqIO.parse(filepath, "fasta"):
        cur_seq = str(record.seq)
        if len(cur_seq) >= min_length:
            mh = sourmash.MinHash(n=0, ksize=31, scaled=10, track_abundance=True)
            mh.add_sequence(cur_seq, force=True) #force=True to allow short sequences and non-ACGT chars
            genomes[record.id] = {
                "record": record,
                "sequence": cur_seq,
                "minhash": mh
            }
    genome_ids = sorted(list(genomes.keys()))
    return genomes, genome_ids

def determine_clusters(filepath: str, genomes: dict, genome_ids: list) -> dict[str]:
    """
    Reads a clustering file and returns an updated version of the genomes dictionary with cluster information.
    """