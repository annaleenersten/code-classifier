# code-classifier

A machine learning project focused on detecting bugs and security vulnerabilities in Python code

## Team Members
- Annalee Nersten
- Tomas Stoker
- Giri Vignesh
- Gregory Shanygin

## Datasets

This project uses two datasets:

1. PyCodeVul 

[vulnerability_dataset.zip](https://github.com/user-attachments/files/34048312/vulnerability_dataset.zip)
   - A benchmark dataset for Python vulnerability detection.
   - Contains code examples labeled or associated with security-related issues.
   - Useful for building and evaluating models that detect vulnerable Python patterns.

2. PyTraceBugs 

[bug_dataset.zip](https://github.com/user-attachments/files/34048293/bug_dataset.zip)
   - A Python bug dataset with trace-based bug examples and debugging patterns.
   - Useful for studying common program defects, runtime issues, and bug classification behavior.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```