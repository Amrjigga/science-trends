# Science Trends Analysis

This project analyzes trends in scientific research using arXiv papers. It clusters millions of papers by topic and shows which research areas are rising or declining over time.

## What it does

- Downloads ~2.87M scientific papers from arXiv via Kaggle
- Cleans the text (titles + abstracts) 
- Creates embeddings using TF-IDF + SVD
- Clusters papers into research topics using k-means
- Generates meaningful topic names like "Large Language Models", "Quantum Information", etc.
- Analyzes trends over time to find rising/declining topics
- Creates interactive visualizations

## Setup

1. **Clone this repo**
```bash
git clone https://github.com/Amrjigga/science-trends.git
cd science-trends
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

3. **Install requirements**
```bash
pip install -r requirements.txt
```

4. **Get Kaggle API key**
- Go to https://www.kaggle.com/account
- Click "Create New API Token" 
- Download the `kaggle.json` file
- Put it in the right place:
  - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
  - Mac/Linux: `~/.kaggle/kaggle.json`
- You might need to create the `.kaggle` folder first

## Running it

Just run this and wait (takes about 30 mins max):
```bash
python src/cli.py
```

## What you'll get

The pipeline creates these folders and files:

### `data/` folder
- `data/raw/arxiv-metadata-oai-snapshot.json` - The raw dataset (1.52GB)
- `data/interim/clean.parquet` - Cleaned paper data
- `data/interim/embeddings.npy` - TF-IDF vectors 
- `data/interim/labels.csv` - Cluster assignments

### `outputs/` folder
- `outputs/tables/topic_names.csv` - Topic names like "Large Language Models"
- `outputs/tables/trends.csv` - Which topics are rising/declining
- `outputs/tables/monthly_counts.csv` - Time series data
- `outputs/tables/cluster_labels.csv` - Raw TF-IDF terms per cluster
- `outputs/figures/top_trends.html` - Interactive chart of rising topics
- `outputs/figures/cluster_size_distribution.html` - How many papers per topic

## Key files explained

- `src/cli.py` - Runs the whole pipeline
- `src/config.yaml` - All the parameters 
- `src/preprocess.py` - Cleans the text data
- `src/embed.py` - Creates TF-IDF embeddings
- `src/cluster.py` - Does k-means clustering
- `src/label_topics.py` - Extracts topic terms
- `src/name_topics.py` - Converts terms to readable names
- `src/aggregate_trends.py` - Analyzes trends over time
- `src/visualize.py` - Makes the charts

## What the results look like

You'll see research topics like:
- **Large Language Models** (rising trend)
- **Quantum Information** (stable)
- **Astrophysics** (multiple clusters for galaxies, dark matter, etc.)
- **Condensed-Matter Physics** (superconductivity, magnetism)
- **String Theory / QFT** (theoretical physics)
- **Deep Learning** (computer vision, neural networks)

The interactive charts show how these topics' popularity changes over time.

## Requirements

- Python 3.8+
- About 8GB RAM 
- Internet connection for downloading the dataset
- Kaggle account 

## Notes

- The `.gitignore` excludes the large data files, so you'll need to run the pipeline to generate them
- First run downloads 1.52GB of data, so make sure you have space
- The clustering finds 10-16 topics automatically using silhouette analysis
- If you want to change parameters, edit `src/config.yaml`

## Running individual steps

If something breaks, you can run steps individually:
```bash
python src/preprocess.py
python src/embed.py
python src/cluster.py
python src/label_topics.py
python src/name_topics.py
python src/aggregate_trends.py
python src/visualize.py
```

Or use the Makefile:
```bash
make all    # run everything
make clean  # delete generated files
```


