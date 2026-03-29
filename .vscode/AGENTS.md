This repo is for NOVA AE mode filtering/classification

RF is feature-based baseline model rf_train_classify.py

cnn_raw, cnn_straightened, and cnn_hybrid are the current CNN family

sort_shot.py is the canonical post-processing entry point

do not hardcode NERSC/Flux absolute paths

preserve feature-schema consistency between training and inference

be careful with mode-array axis ordering and flattening conventions

I am still trying to organize this repo, paths to data and scripts etc