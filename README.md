## How to install prerequisites
Run these commands in order:
```shell
pip install -r packages/requirements.txt
python nltk_requirement_installation.py
```

## Write indexer to file
Assuming data has this structure:
```shell
.
└── search_engine
    ├── data
        ├── ANALYST
        └── DEV
    ├── packages
    └── indexer.py
    └── m1_report.py
    └── ...
```
```shell
# default for analyst dataset (for faster testing)
python run.py

# for dev dataset, can change data argument to any path
python run.py --data data/DEV
```

## Milestone 1 report
Get milestone 1 report after obtaining doc_id.pkl and inverted_index.pkl
```shell
python m1_report.py
```