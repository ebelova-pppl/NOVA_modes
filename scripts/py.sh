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

# To view the results, you can use the following command:
python viz/view_modes_csv.py "$SHOT/mode_labels_clean.csv" --topk 100 