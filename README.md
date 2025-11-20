# Tool


Example steps for downloading data from NCBI

* DOWNLOADING DATA
Using NCBI datasets (https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/):
- downloading genomes: 
```bash
datasets download genome taxon TAXID --assembly-level complete --assembly-source GenBank --exclude-atypical --filename FILENAME
```
This downloads all "Complete Genome" quality genomes from GenBank (this excludes duplicates that may also appear in RefSeq) while excluding genomes listed as "atypical" (which are usually hidden when browsing NCBI Genomes).
- downloading metadata:
```bash
datasets summary genome taxon TAXID --assembly-level complete --assembly-source GenBank --exclude-atypical --as-json-lines | dataformat tsv genome --fields accession,organism-tax-id,source_database,assminfo-atypicalis-atypical > metadata.tsv
```
This downloads the metadata (primarily the taxid) for the genomes downloaded through the previous command and stores it in a tsv file called `metadata.tsv`.

## FLP-based model
### Parameters
$$
\begin{array}{ll}
G &\text{set of all genomes} \\
t: G \rightarrow \mathbb{N} &\text{taxonomic mapping} \\
\sim_t &\text{equivalence relation for all genomes based on taxonomic mapping} \\
\{ {T_1, T_2, \ldots}, T_m \} =: G \setminus \sim_t &\text{set of all taxonomic classes} \\
s: G \times G \rightarrow [0,1] \cap \mathbb{Q} &\text{similarity mapping between genomes} \\
f: g \rightarrow \mathbb{R}_{\geq 0} &\text{cost for selecting a genome (generally constant)}
\end{array}
$$

### Decision Variables
$$
\begin{array}{llr}
x_g &= 
\begin{cases}
    1, & \text{if genome } g \text{ is selected} \\
    0, & \text{otherwise}
\end{cases} &\qquad \forall g \in G\\
y_{g,g'} &= 
\begin{cases}
    1, & \text{if genome } g' \text{ is selected as representative for } g \\
    0, & \text{otherwise}
\end{cases} &\qquad \forall g \sim_{t} g' \\
z_{g,g'} &=
\begin{cases}
    1, & \text{if genome } g \text{ and } g' \text{ are both selected} \\
    0, & \text{otherwise}
\end{cases} &\qquad \forall g \not\sim_{t} g' \\
q_{i,j} &\geq 0 \text{ penalty for distance between closest selected genomes for } T_i \text{ and } T_j \text{} &\qquad \forall T_i \neq T_j
\end{array}
$$

### Objective function
$$
\begin{array}{llr}
\text{Minimize} & \sum_{g \in G}f(g)x_g + \sum_{g \sim_t g'} \left(1 - s(g,g')\right)y_{g,g'} + \sum_{i \neq j}q_{T_i, T_j} &(1)
\end{array}
$$

### Constraints
$$
\begin{array}{lllc}
\sum_{g' \sim_t g}y_{g,g'} & \geq 1 & \forall g \in G &(2) \\
y_{g,g'} & \leq x_{g'} & \forall g \sim_t g' &(3) \\
z_{g,g'} & \geq -1 + \left(x_g + x_{g'}\right) & \forall g \not\sim_t g' &(4a) \\
z_{g,g'} & \leq 0.5\left(x_g + x_{g'}\right) & \forall g \not\sim_t g' &(4b) \\
q_{g,g'} & \geq s(g,g')z_{g,g'} & \forall g \not\sim_t g' &(5) \\
x_g \in \{0,1\} & & \forall g \in G &(6) \\
y_{g,g'} \in \{0,1\} & & \forall g \sim_t g' &(7) \\
z_{g,g'} \in \{0,1\} & & \forall g \not\sim_t g' &(8) \\
q_{T_i, T_j} \geq 0 & & \forall i \neq j &(9)
\end{array}
$$
