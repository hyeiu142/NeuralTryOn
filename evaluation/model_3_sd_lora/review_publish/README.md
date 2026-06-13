# Published Result Review

These stages inspect a previously published unpaired holdout Kaggle Dataset.
They do not require loading the SD + LoRA checkpoint.

```text
01_review_published_dataset.py  Browse and rank all published results
02_review_manual_shortlist.py   Inspect selected person-cloth pairs
03_export_shortlist_gallery.py  Export report-ready three-column pages
```

Edit the dataset slug, page, sort field, or shortlist directly at the beginning
of each script before running it.
