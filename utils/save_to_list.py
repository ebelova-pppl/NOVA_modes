import glob

# saving bad and good modes for training
files_b = glob.glob("BAD/egn*")
files_g = glob.glob("GOOD/egn*")

with open("train_list.csv","w") as f:
    for p in files_b:
        f.write(f"{p},bad\n")
with open("train_list.csv","a") as f:
    for p in files_g:
        f.write(f"{p},good\n")
