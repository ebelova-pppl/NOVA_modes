# tcsh - commands to run the mode labeling script on the whole shot N1-10 

set SHOT  = /p/hym/ebelova/NOVA/data_mixed/nstxuG142301W29
set SPLIT = /p/hym/ebelova/NOVA/data_mixed/nstxuG142301W29_tae_eae_split
set OUT   = "$SHOT/mode_labels_human_review.csv"
foreach n (1 2 3 4 5 6 7 8 9 10)
    python scripts/label_modes_fast.py "$SHOT/N$n" \
        --mode-list "$SPLIT/tae_like.csv" \
        --csv_out "$OUT" --no-rf
endforeach


# same for bash:

SHOT=/p/hym/ebelova/NOVA/data_mixed/nstxuG121123B12
SPLIT=/p/hym/ebelova/NOVA/data_mixed/nstxuG121123B12_tae_eae_split
OUT="$SHOT/mode_labels_unchanged_may_review.csv"

for n in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/label_modes_fast.py "$SHOT/N$n" \
    --mode-list "$SPLIT/tae_like_unchanged_may_labels_to_review.csv" \
    --csv_out "$OUT" --no-rf
done

# On Perlmutter:

SHOT="$NOVA_DATA/nstxuE205052A01t022"
SPLIT="$NOVA_REPO/tests/labels_audit"
OUT="$NOVA_REPO/tests/labels_audit/labels_human_review.csv"

for n in 1 2 3 4 5 6 7 8 9 10; do
  python scripts/label_modes_fast.py "$SHOT/N$n" \
    --mode-list "$SPLIT/tae_like_audit.csv" \
    --csv_out "$OUT" --no-rf
done

# To view the results, you can use the following command:
python viz/view_modes_csv.py "$SHOT/mode_labels_clean.csv" --topk 100 

#########################
# To run GPU LOSO test on Perlmutter (with continuum option) use:

salloc -A m314_g -C gpu -q interactive -N 1 \
  --gpus=1 --cpus-per-task=1 -t 04:00:00
  cd "$NOVA_REPO"
module load pytorch
source configs/paths/nova_paths.nersc.sh
python -u scripts/run_loso_10.py \
  --steps all \
  --train_csv "$NOVA_TRAIN_CSV" \
  --out_root outputs/loso_15_raw_continuum_branch_M100_bs8 \
  --work_root "$SCRATCH/nova_s/loso_15_raw_continuum_branch_M100_bs8" \
  --cnn_launch srun \
  --cnn_device cuda \
  --sort_device cpu \
  --cnn_batch_size 8 \
  --cnn_m_target 100 \
  --cnn_continuum_branch \
  --cnn_cache_data \
  --cnn_refit_full_before_save


# To train RF on Perlmutter, use the following command:
source configs/paths/nova_paths.nersc.sh
nova_env

python "$NOVA_REPO/scripts/rf_train_classify.py" \
  --train_csv "$NOVA_TRAIN_CSV" \
  --model_out "$NOVA_REPO/models/nova_mode_classifier.joblib"

# To train cnn_raw on Perlmutter, use the following command:
cd /global/homes/e/ebelova/src_nova
module load pytorch
source configs/paths/nova_paths.nersc.sh
nova_env

python $NOVA_REPO/scripts/cnn_raw.py \
  --train_csv $NOVA_TRAIN_CSV \
  --data_dir $NOVA_DATA \
  --refit_full_before_save \
  --model_out $NOVA_REPO/models/nova_cnn_raw.pt \
  --M_target 100 \
  --batch_size 8 \
  --cache_data
