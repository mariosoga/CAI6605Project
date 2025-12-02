Set-Location -Path $PSScriptRoot

$epsilons = "0.000","0.050","0.100","0.150","0.200","0.300"

foreach ($eps in $epsilons) {
    Write-Host "🚀 Starting training for epsilon $eps"
    python scripts/train_and_eval_model.py `
        --data_dir "data/dataset/augmented_datasets/augmented_eps_$eps" `
        --out_dir "outputs/fsgmRun/eps_$eps" `
        --epochs 20 `
        --multitask `
        --healthy_keyword healthy `
        --health_loss_weight 1.0 `
        --save_hard_examples --hard_k 50 `
        --save_confusion_pairs --pairs_m 5 --examples_per_pair 8 `
        --grad_cam --grad_cam_k 50 --grad_cam_task disease
    Write-Host "✅ Finished training for epsilon $eps"
}