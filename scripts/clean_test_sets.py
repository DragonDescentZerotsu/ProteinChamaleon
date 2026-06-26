"""
Remove test examples whose sequences appear in train or val sets.
Saves cleaned NPZ files alongside originals with _clean suffix.

Usage:
    python scripts/clean_test_sets.py
"""

import numpy as np
from pathlib import Path

ALIGN_DIR  = Path("/data2/steven/data/stage2/alignment")
INTER_DIR  = Path("/data2/steven/data/stage2")


def clean_npz(test_path: Path, train_path: Path, val_path: Path):
    print(f"\nCleaning {test_path.name}...")

    test  = np.load(test_path,  allow_pickle=True)
    train = np.load(train_path, allow_pickle=True)
    val   = np.load(val_path,   allow_pickle=True)

    contaminated_seqs = (
        set(str(s) for s in train["sequences"]) |
        set(str(s) for s in val["sequences"])
    )

    test_seqs = [str(s) for s in test["sequences"]]
    keep = [i for i, s in enumerate(test_seqs) if s not in contaminated_seqs]
    remove = len(test_seqs) - len(keep)

    print(f"  Original : {len(test_seqs)}")
    print(f"  Removed  : {remove} ({100*remove/len(test_seqs):.1f}%)")
    print(f"  Remaining: {len(keep)}")

    # Build cleaned arrays for every key in the NPZ
    cleaned = {}
    for key in test.files:
        arr = test[key]
        if len(arr) == len(test_seqs):
            cleaned[key] = arr[keep]
        else:
            cleaned[key] = arr  # e.g. scalar metadata

    out_path = test_path.parent / test_path.name.replace(".npz", "_clean.npz")
    np.savez(out_path, **cleaned)
    print(f"  Saved to : {out_path}")
    return len(keep), remove


# Alignment
a_kept, a_removed = clean_npz(
    ALIGN_DIR / "alignment_test.npz",
    ALIGN_DIR / "alignment_train.npz",
    ALIGN_DIR / "alignment_val.npz",
)

# Interleaved
i_kept, i_removed = clean_npz(
    INTER_DIR / "interleaved_test.npz",
    INTER_DIR / "interleaved_train.npz",
    INTER_DIR / "interleaved_val.npz",
)

print(f"""
=== Summary ===
Alignment  test: {a_kept} clean examples ({a_removed} removed)
Interleaved test: {i_kept} clean examples ({i_removed} removed)

Clean files saved as:
  {ALIGN_DIR}/alignment_test_clean.npz
  {INTER_DIR}/interleaved_test_clean.npz

Update run_eval.sh to use these paths, or pass --align-test / --interleaved-test flags.
""")
