## How to install prerequisites
Run these commands in order:
```shell
pip install -r packages/requirements.txt
python packages/nltk_requirement_installation.py
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
python indexer.py

# for dev dataset, can change data argument to any path
python indexer.py --data data/DEV
```

## Simple terminal search engine
Assuming inverted index and doc id map is obtained
```shell
python run.py
```

## Milestone 1 report
Get milestone 1 report after obtaining doc_id.pkl and inverted_index.pkl
```shell
python m1_report.py
```

## Milestone 2 report
Assuming inverted index and doc id map is obtained
```shell
python m2_report.py
```
