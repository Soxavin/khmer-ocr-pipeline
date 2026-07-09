# Dataset-factory corpus (local only — PDFs are gitignored, never committed)

Drop collected source PDFs here, one subfolder per source. This folder feeds the
Week-2 dataset factory (`datagen/pseudo_label_layout.py`, `datagen/harvest_table_gt.py`);
the eval ground truth stays separate in `eval/datasets/real/` (also local-only).

| Folder | Source | Notes |
|---|---|---|
| `ardb_daily/` | ARDB daily market-price bulletins | born-digital Unicode → feeds all 3 dataset products |
| `moc_gas/` | MoC gas-price bulletins | classify on arrival (run the helper below) |
| `budget_tofe/` | Budget-execution TOFE reports | Khmer text layer is legacy mojibake — numbers-only harvest |
| `other/` | anything else in scope | classify before routing |

Target: **≥40 docs / ≥100 pages** total. Check progress + classification any time:

```bash
uv run python scripts/collect_documents.py corpus/
# or download from a URL list first (one URL per line):
uv run python scripts/collect_documents.py corpus/ardb_daily/ --urls urls.txt
```

Heed the `khmer_layer_suspect` warnings in the report: those docs' numbers are usable
(PyMuPDF `find_tables`) but their Khmer text must never be used as free recognition GT.
Copy the existing `sample_data/` docs in here too so the corpus is self-contained
(same filename = same doc; the factory will treat filenames as document IDs).
