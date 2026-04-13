### train_master.csv 
this is the original training list for TAE dataset
it includes relative file paths and good/bad labels.
Paths are relative as nstx_123456/N1/egn*

### train_master_full_paths.csv
same as train_master, but with absolute paths

### train_tae.csv
same list as train_master.csv, but include second set of labels (none or tae)
ie each record includes: path,validity,family for TAE modes

### all_modes.csv
New extended, complete and verified training list.
Includes a header row: path,validity,family
for all modes, TAEs and EAEs.
validity: good or bad
family: tae or eae

### good_tae.csv
New complete and verified list of good TAEs

### good_eae.csv
New complete and verified list of good EAEs
