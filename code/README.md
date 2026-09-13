# CPST-DDI: AAAI Reproducibility Code

This folder contains the minimal implementation required to reproduce the
CPST-DDI model, its two independently trained experts, and the main ablations.
Run all commands from this `code` directory.

## Directory layout

```text
AAAI-code/
├── code/
│   ├── main.py
│   ├── audit_leakage.py
│   ├── prepare_data.py
│   ├── run_cv.py
│   ├── ensemble.py
│   ├── configs.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── metrics.py
├── dataset/
│   └── <dataset>/
│       ├── ddi.txt
│       ├── drug_smiles.txt
│       └── networks.txt
├── cache/
└── results/
```

`ddi.txt` has four integer columns:
`drug_a drug_b candidate_relation binary_label`. `networks.txt` starts with an
edge-count line and then contains `source target auxiliary_relation`.

## Environment

```bash
conda create -n cpst-ddi python=3.10 -y
conda activate cpst-ddi
pip install -r requirements.txt
```

CUDA is used automatically when available and supported by the installed
PyTorch build. For recent GPUs, install a PyTorch wheel whose compiled CUDA
architectures include the device compute capability; otherwise the code safely
falls back to CPU.

## Leakage-safe protocol

The release code applies the following safeguards by default:

1. All rows sharing the same unordered drug pair are assigned to the same
   outer fold, including reverse pairs and different candidate relations.
2. The inner validation split is also grouped by unordered drug pair.
3. Pair-topology features are built only from positive edges in the current
   training split.
4. The queried edge is masked when computing topology features, preventing a
   training positive from contributing its own edge to degree statistics.
5. All auxiliary-KG edges whose two endpoints are DDI drugs are excluded,
   preventing a target drug pair from appearing directly as a KG neighbor.
6. Expert ensembling checks labels, drug pairs, candidate relations, and
   original sample indices before averaging probabilities.
7. Test metrics are computed once from the validation-selected checkpoint.

Drug ID embeddings are transductive identity features, not a claim of
generalization to unseen drugs. Use the `no_drug_id` ablation or a separate
drug-disjoint protocol when evaluating cold-start performance.

Run the audit before training:

```bash
python main.py audit
```

The report is written to `../results/leakage_audit.json`.

## Data preparation

```bash
python main.py prepare
```

Only datasets present under `../dataset` are processed. Molecular and
auxiliary-KG features are deterministic and cached under `../cache`.

## Main two-expert experiment

```bash
python main.py main
```

This independently trains:

- `structure_expert`: CPST without fold-local pair topology;
- `topology_expert`: the same CPST backbone with fold-local pair topology.

The final probability is the fixed average:

```text
p_final = 0.5 * p_structure + 0.5 * p_topology
```

No test labels are used to select the ensemble weight.

## Strict drug-disjoint cold start

The following setting retains only rows for which both endpoints belong to
the same drug partition.  Test drugs are absent from both training and
validation, and crossing rows are discarded.  Drug ID and DDI-pair topology
are disabled because they are unavailable for unseen drugs.

```bash
python run_cv.py --dataset kegg --variant cold_start \
  --tag drug_disjoint_cold_start --set split_mode=drug_disjoint
```

S2 (exactly one unseen endpoint) uses the same leakage-safe inductive model:

```bash
python run_cv.py --dataset kegg --variant cold_start_inductive \
  --tag s2_one_unseen --set split_mode=one_unseen
```

For the strict S3 setting, `cold_start_inductive` additionally removes raw KG
neighbor-identity embeddings while retaining transferable KG relation types and
edge directions.  This option must be selected by validation behavior, not by
test-set metrics.

## Ablations

```bash
python main.py ablation
```

The included variants are:

- `no_kg`: no auxiliary KG and no transport;
- `concat`: pooled KG only, without transport;
- `partial_ot`: semantic partial transport without structural cost or
  counterfactual loss;
- `no_counterfactual`: full structure-aware transport without keep/remove
  losses;
- `no_drug_id`: remove the transductive Drug ID embedding.

For a short smoke test:

```bash
python run_cv.py --dataset kegg --variant structure_expert \
  --tag smoke --set folds=2 epochs=1 patience=1 batch_size=64
```

Each result directory contains the configuration, split indices, validation
history, selected checkpoint, per-sample predictions, and `metrics.csv` with
fold-wise values, population mean, and population standard deviation rounded
to four decimal places.

## Important reproducibility note

This leakage-safe pair-grouped protocol is stricter than row-wise stratified
cross-validation. Consequently, its scores should not be presented as the same
experimental protocol as results produced by row-wise splitting. Rerun all
main models, ablations, and baselines under this protocol before replacing
previous paper tables.
