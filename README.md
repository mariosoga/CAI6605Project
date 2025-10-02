## Single-task (disease only)
```bash
python pipeline.py --out_dir outputs --epochs 20
```

## Multitask (disease + healthy/sick)
Health label is derived by checking if the class name contains `--healthy_keyword` (default: "healthy").
```bash
python pipeline.py --out_dir outputs --epochs 20 --multitask --healthy_keyword healthy --health_loss_weight 0.5 --save_hard_examples --hard_k 50 --save_confusion_pairs --pairs_m 5
```

**Artifacts**
- Disease: `classification_report_disease.txt`, `confusion_matrix.png`, `val_predictions_disease.csv`
- Health:  `classification_report_health.txt`, `val_predictions_health.csv`
- Hard examples and confusion galleries
- Best checkpoint: `best.pt` 
