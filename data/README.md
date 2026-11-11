# Data

This folder holds the raw datasets and the cleaned versions used to train two binary classifiers on Python code:

1. **Bug classifier** - is a code sample buggy (1) or correct (0)?
2. **Vulnerability classifier** - is a function vulnerable (1) or non-vulnerable (0)?

## Dataset sources

| Task | Dataset | Source | Description |
|---|---|---|---|
| Vulnerability | PyCode_Vul | [Kaggle](https://www.kaggle.com/datasets/oulabspring/pycode-vul-a-benchmark-dataset-for-python?resource=download) | Python functions labeled vulnerable or non-vulnerable, with repository and commit metadata. Provided as a train file and a test file. |
| Bugs | PyTraceBugs | [Hugging Face](https://huggingface.co/datasets/gurudesh/pytracebugs/tree/main) | Pairs of buggy and fixed Python code taken from GitHub bug-fix commits, with the traceback that the bug caused. Provided as train, valid and test CSVs. |


## Raw data format

### PyTraceBugs (`bugfixes_*.csv`): one row per buggy/fixed pair

| Column | Meaning |
|---|---|
| `before_merge` | the buggy code |
| `after_merge` | the fixed code |
| `url` | GitHub issue the bug was reported in |
| `full_traceback` | the error output caused by the bug |
| `traceback_type` | error type, e.g. `TypeError` |

Row counts: train 14,118 | valid 9,457 | test 161 (23,736 pairs in total).

Example of one raw row (code abridged):

| before_merge | after_merge | url | traceback_type |
|---|---|---|---|
| `def tf_dtype(np_dtype): try: ... except Exception as e: return tf.variant` | same function, with `logger.log(e)` added in the `except` block | `https://github.com/<owner>/<repo>/issues/<n>` | `TypeError` |

### PyCode_Vul (`PyCode_Vul-_train-set.csv`, `PyCode_Vul_test-set.csv`): one row per function

| Column (train) | Column (test) | Meaning |
|---|---|---|
| `vulnerable_function_source` | `function_code` | the function's source code |
| `label` | `class` | 1 = vulnerable, 0 = non-vulnerable |
| `patched_function_source` | - | the patched version of the function |
| `repo`, `sha`, `file_path` | same | where the function came from |
| `__source_file`, `cwe_ids`, `cve_ids`, `predicted_cwe_ids` | same | extra metadata (see "Known issues") |

Row counts: train 14,248 (6,298 vulnerable, 7,950 non-vulnerable) | test 3,563 (1,601 vulnerable, 1,962 non-vulnerable).

Note that the two files name the same columns differently.

## Processed data format

Both tasks are converted to the same shape, one code sample per row:

| Column | Meaning |
|---|---|
| `code` | the code text, which is the only input to the models |
| `label` | 1 = buggy / vulnerable, 0 = correct / non-vulnerable |
| `group`, `pair_id` | bugs only: repository name and pair ID, used to keep splits fair |
| `repo`, `sha`, `file_path` | vulnerability only: kept for diagnostics, never used as features |

### Bug data: each raw pair becomes two rows

Raw (1 row):

| before_merge | after_merge | url |
|---|---|---|
| `def tf_dtype(...): ... return tf.variant` | `def tf_dtype(...): ... logger.log(e) ... return tf.variant` | `.../<owner>/<repo>/issues/<n>` |

Processed (2 rows):

| code | label | pair_id | group |
|---|---|---|---|
| buggy version (`before_merge`) | 1 | 0 | `<owner>/<repo>` |
| fixed version (`after_merge`) | 0 | 0 | `<owner>/<repo>` |

### Vulnerability data: columns are unified and trimmed

Raw (train and test have different names, plus metadata columns):

| vulnerable_function_source | label | repo | sha | file_path | `__source_file` | predicted_cwe_ids |
|---|---|---|---|---|---|---|
| `def ...` | 1 | `<repo>` | `<sha>` | `<path>` | *(empty)* | *(filled in)* |

Processed:

| code | label | repo | sha | file_path |
|---|---|---|---|---|
| `def ...` | 1 | `<repo>` | `<sha>` | `<path>` |

The `__source_file`, `predicted_cwe_ids`, `cwe_ids` and `cve_ids` columns are dropped.

## Why the data is processed

| Step | Reason |
|---|---|
| Split each bug pair into two labeled rows | a classifier needs one input and one label per row |
| Unify the vulnerability train/test column names | the two files use different names, which breaks a shared loader |
| Drop leaky columns | they reveal the label without the model reading any code |
| Drop pairs where buggy equals fixed, and drop duplicate rows | no bug to learn from, and duplicates inflate scores |
| Pool the three bug files and re-split by repository | the provided bug test file overlaps heavily with the validation file; splitting by repo keeps near-identical code out of both train and test |

Resulting bug data: 23,438 pairs from 630 repositories (after dropping identical and duplicate pairs), giving 46,876 rows split by repository:

| Split | Rows | Repos | Buggy | Correct |
|---|---|---|---|---|
| Train | 32,626 | 440 | 16,313 | 16,313 |
| Validation | 6,588 | 95 | 3,294 | 3,294 |
| Test | 7,662 | 95 | 3,831 | 3,831 |

The vulnerability data keeps the provided train/test split, with exact duplicates and rows with missing code removed (a small number, about 60 in train).


