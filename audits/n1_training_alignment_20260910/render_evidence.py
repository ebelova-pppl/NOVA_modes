"""Render signed-profile/log/continuum comparisons for explicitly listed modes.

Example: python audits/n1_training_alignment_20260910/render_evidence.py \
  --audit-dir outputs/review_n1_training_alignment_20260910 \
  --mode-list audits/n1_training_alignment_20260910/review_modes.csv
"""
import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/"src"),str(REPO/"scripts")]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cont_features import continuum_crossing_records, load_datcon_for_mode
from nova_mode_loader import load_mode_from_nova


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir",required=True,type=Path)
    parser.add_argument("--mode-list",required=True,type=Path)
    args=parser.parse_args()
    metadata=json.loads((args.audit_dir/"metadata.json").read_text())
    data=Path(metadata["data_root"])
    with (args.audit_dir/"mode_coverage.csv").open() as handle:
        coverage={r["path"]:r for r in csv.DictReader(handle)}
    with args.mode_list.open() as handle:
        keys=[r["path"] for r in csv.DictReader(handle)]
    out=args.audit_dir/"figures";out.mkdir(exist_ok=True)
    for shot in sorted({key.split("/")[0] for key in keys}):
        selected=[key for key in keys if key.split("/")[0]==shot]
        fig,axes=plt.subplots(2,len(selected),figsize=(5*len(selected),6),squeeze=False,sharex="col")
        for col,key in enumerate(selected):
            row=coverage[key]
            mode,omega,*_=load_mode_from_nova(str(data/key))
            r=np.linspace(0,1,mode.shape[1])
            low,high,*_=load_datcon_for_mode(str(data/key),len(r))
            crossings=continuum_crossing_records(mode,omega,low,high)
            # Partial log blocks must not look like complete resonance evidence.
            singularities=json.loads(row["singularities"]) if row["status"]=="MATCHED" else []
            top,bottom=axes[:,col]
            top.plot(r,np.sqrt(low),label="Lower continuum",color="tab:blue")
            top.plot(r,np.sqrt(high),label="Upper continuum",color="tab:orange")
            top.axhline(omega,color="black",label="Mode frequency")
            top.set_ylim(0,1.5*omega)
            title=key.split("/",1)[1]
            if row["status"]!="MATCHED":
                title += "\n" + row["status"]
            top.set_title(title,fontsize=9)
            top.set_ylabel("Frequency (NOVA units)")
            bottom.plot(r,mode.T,lw=.8,alpha=.85)
            bottom.set_ylim(-1.05,1.05)
            bottom.set_ylabel("Signed harmonic amplitude")
            bottom.set_xlabel("Normalized radius")
            for ax in (top,bottom):
                for j,crossing in enumerate(crossings):
                    ax.axvline(crossing["r_cross"],color="crimson",lw=1.2,alpha=.8,
                               label="Datcon crossing" if j==0 else None)
                for j,radius in enumerate(singularities):
                    ax.axvline(radius,color="black",ls="--",lw=.8,alpha=.65,
                               label="Logged singularity" if j==0 else None)
                ax.set_xlim(0,1);ax.grid(alpha=.15)
            if col==0:
                top.legend(fontsize=7,loc="upper left")
                bottom.legend(fontsize=7,loc="lower left")
        fig.suptitle(f"{shot}\nContinuum crossings, NOVA log and signed profiles",fontsize=11)
        fig.tight_layout()
        fig.savefig(out/f"{shot}.png",dpi=140)
        plt.close(fig)


if __name__=="__main__":
    main()
