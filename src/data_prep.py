import re  # regular expressions, used to pull the repo name out of a GitHub URL

import pandas as pd  # tables (DataFrames) and CSV reading/writing
from sklearn.model_selection import GroupShuffleSplit  # splits data by group

# Fixed random seed so everyone on the team gets the exact same split.
SEED = 42

# --- Vulnerability data paths ---
VULN_TRAIN_PATH = "data/raw/PyCode_Vul-train-set.csv"
VULN_TEST_PATH = "data/raw/PyCode_Vul-test-set.csv"
# Columns we keep ONLY to inspect the data, never to train a model.
META = ["repo", "sha", "file_path"]

# --- Bug data paths ---
BUG_FILES = ["data/raw/bugfixes_train.csv",
             "data/raw/bugfixes_valid.csv",
             "data/raw/bugfixes_test.csv"]


# =======================================================================
# 1. VULNERABILITY DATA
# =======================================================================
# Load, clean, and return (train, test) DataFrames for the vulnerability task.
def load_vuln(train_path=VULN_TRAIN_PATH, test_path=VULN_TEST_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # The two files name their columns differently, so give them the same names:
    #   train: "vulnerable_function_source" -> "code"
    #   test:  "function_code" -> "code" and "class" -> "label"
    train = train.rename(columns={"vulnerable_function_source": "code"})
    test = test.rename(columns={"function_code": "code", "class": "label"})

    # Keep only the code, the label, and the diagnostic metadata. This also
    # drops the leaky columns listed at the top of this file.
    train = train[["code", "label"] + META]
    test = test[["code", "label"] + META]

    # Remember the original sizes so we can report how many rows were removed.
    n_train, n_test = len(train), len(test)

    # Drop rows with no code at all, then drop exact duplicate functions.
    # (Duplicates inflate scores: the same function could be in train and test.)
    train = train.dropna(subset=["code"]).drop_duplicates(subset="code")
    test = test.dropna(subset=["code"]).drop_duplicates(subset="code")

    # Remove any training function that also appears in the test set. None were
    # found in the current data, but this guards against it if the data changes.
    train = train[~train["code"].isin(set(test["code"]))]

    # Print a summary so we can check the cleaning worked as expected.
    print(f"[vuln] train: {n_train} -> {len(train)} rows after cleaning")
    print(f"[vuln] test:  {n_test} -> {len(test)} rows after cleaning")
    print("[vuln] train label counts:", train["label"].value_counts().to_dict())
    print("[vuln] test label counts: ", test["label"].value_counts().to_dict())

    # Warning sign: if the same commit appears in train AND test, related
    # functions may be on both sides, which can make test scores look too good.
    shared = len(set(train["sha"]) & set(test["sha"]))
    print(f"[vuln] commit shas shared between train and test: {shared} (possible leakage)")

    # reset_index renumbers the rows 0,1,2,... after rows were removed.
    return train.reset_index(drop=True), test.reset_index(drop=True)


# =======================================================================
# 2. BUG DATA
# =======================================================================
# Turn 'https://github.com/owner/repo/issues/12' into 'owner/repo'.
# We use the repo name as the 'group' so that all code from one repository stays together in a single split.
def repo_from_url(url):
    m = re.search(r"github\.com/([^/]+/[^/]+)/", str(url))
    # If the URL isn't a normal GitHub link, fall back to the raw URL text.
    return m.group(1).lower() if m else str(url)

# Load the three bug CSVs, pool them, and clean the buggy/fixed pairs.
def load_pairs(files=BUG_FILES):
    # Stack train + valid + test into one big table (we re-split later).
    pairs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    n0 = len(pairs)  # original number of pairs, for the printout below

    # Remove pairs missing either version of the code.
    pairs = pairs.dropna(subset=["before_merge", "after_merge"])

    # Remove pairs where "buggy" and "fixed" are identical: there is no bug
    # to learn from, and they would add contradictory examples.
    same = pairs["before_merge"] == pairs["after_merge"]
    # Then remove exact duplicate pairs (same buggy AND fixed code).
    pairs = pairs[~same].drop_duplicates(subset=["before_merge", "after_merge"])
    print(f"[bugs] pairs: {n0} -> {len(pairs)} (dropped identical and duplicate pairs)")

    pairs = pairs.reset_index(drop=True)
    # Give each pair a unique ID so we can match a buggy row to its fixed row.
    pairs["pair_id"] = pairs.index
    # The group is the repository; used later to keep repos from being split up.
    pairs["group"] = pairs["url"].map(repo_from_url)
    print("[bugs] unique repos:", pairs["group"].nunique())
    return pairs

# Turn each (buggy, fixed) pair into two labeled rows.
def pairs_to_rows(pairs):
    # Buggy version of every pair -> label 1
    buggy = pd.DataFrame({"code": pairs["before_merge"], "label": 1,
                          "pair_id": pairs["pair_id"], "group": pairs["group"]})
    # Fixed version of every pair -> label 0
    fixed = pd.DataFrame({"code": pairs["after_merge"], "label": 0,
                          "pair_id": pairs["pair_id"], "group": pairs["group"]})
    # Stack them: N pairs become 2N rows, perfectly balanced 50/50.
    return pd.concat([buggy, fixed], ignore_index=True)

# Split into train/val/test so that no repository appears in more than one.
# A normal random split could put a buggy function in train and its nearly identical fixed twin in test, 
# so the model would seem to perform bettermthan it really does. Splitting by repo prevents that.
def split_by_group(rows, val_size=0.15, test_size=0.15, seed=SEED):
    # Step 1: carve off the test set (15% of the repos).
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(rows, groups=rows["group"]))
    trainval, test = rows.iloc[trainval_idx], rows.iloc[test_idx]

    # Step 2: carve the validation set out of what's left. We ask for a larger
    # fraction here (0.15 / 0.85) because it is a share of the remaining 85%,
    # so that validation ends up being 15% of the whole dataset.
    rel_val = val_size / (1 - test_size)
    gss = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(gss.split(trainval, groups=trainval["group"]))
    train, val = trainval.iloc[train_idx], trainval.iloc[val_idx]

    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))

# Full bug pipeline: load -> clean -> make rows -> split. Returns (train, val, test).
def load_bugs():
    rows = pairs_to_rows(load_pairs())
    train, val, test = split_by_group(rows)

    # Print the size, number of repos, and label balance of each split.
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"[bugs] {name}: {len(d)} rows, {d['group'].nunique()} repos, "
              f"label counts {d['label'].value_counts().to_dict()}")

    # Safety checks: crash loudly if any repo ended up in two splits.
    assert not set(train["group"]) & set(test["group"])
    assert not set(train["group"]) & set(val["group"])
    return train, val, test


# =======================================================================
# Running this file directly (python data_prep.py) saves the cleaned data.
# =======================================================================
if __name__ == "__main__":
    # Vulnerability data -> data/processed/vuln_*.csv
    vtr, vte = load_vuln()
    vtr.to_csv("data/processed/vuln_train.csv", index=False)
    vte.to_csv("data/processed/vuln_test.csv", index=False)

    # Bug data -> data/processed/bugs_*.csv
    btr, bva, bte = load_bugs()
    btr.to_csv("data/processed/bugs_train.csv", index=False)
    bva.to_csv("data/processed/bugs_val.csv", index=False)
    bte.to_csv("data/processed/bugs_test.csv", index=False)

    print("saved to data/processed/")