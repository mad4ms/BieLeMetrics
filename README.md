
# BieLeMetrics (Bielefeld Lemgo Metrics)

BieLeMetrics is a Python-based project aimed at processing handball data from Kinexon and Sportradar sources. The project includes downloading, synchronizing, and extracting features from event data to train machine learning models, such as an expected goal model, using the MLJAR platform.

This repository contains the official code implementation for the paper  [*"Expected Goals Prediction in Professional Handball using Synchronized Event and Positional Data"*](https://dl.acm.org/doi/10.1145/3606038.3616152) by the original authors.


![Demo GIF](./assets/events/videos/demo.gif)


## Table of Contents
- [BieLeMetrics (Bielefeld Lemgo Metrics)](#bielemetrics-bielefeld-lemgo-metrics)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Usage](#usage)
    - [Downloading Data](#downloading-data)
    - [Processing Data](#processing-data)
    - [Feature Extraction](#feature-extraction)
    - [Training Machine Learning Model](#training-machine-learning-model)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

The goal of BieLeMetrics is to provide a seamless and automated pipeline to:
1. **Download** data from Sportradar and Kinexon.
2. **Process** the data by synchronizing event information between sources.
3. **Extract** features to be used for machine learning tasks, such as training an expected goal model using MLJAR.

## Features

- Download game data from Sportradar and Kinexon sources.
- Synchronized data processing to align events across different sources.
- Feature extraction and CSV output for MLJAR-based model training.
- Parallel processing capabilities for efficient data handling.

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Required Python packages: declared in `pyproject.toml`

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BieLeMetrics.git
   cd BieLeMetrics
   ```

2. Initialize:
   ```bash
   uv sync --no-dev
   ```

3. Configure environment variables:
   - Create a `.env` file in the root directory and set up necessary environment variables for your Kinexon and Sportradar API keys. Behold, the needed variables are (I must emphasize that I do not have any influence on this login procedure):

   ```bash
   # Kinexon Session Endpoint
    ENDPOINT_KINEXON_SESSION=""
    # Kinexon Main Endpoint
    ENDPOINT_KINEXON_MAIN=""
    # Kinexon API Endpoint
    ENDPOINT_KINEXON_API=""
    # Kinexon Session Username
    USERNAME_KINEXON_SESSION=""
    # Kinexon Main Username
    USERNAME_KINEXON_MAIN=""
    # Kinexon Session Password
    PASSWORD_KINEXON_SESSION=""
    # Kinexon Main Password
    PASSWORD_KINEXON_MAIN=""
    # Kinexon API Key
    API_KEY_KINEXON=""
    # Sportradar API Key
    API_KEY_SPORTRADAR=""
    # Nextcloud Storage Endpoint
    ENDPOINT_STORAGE_NEXTCLOUD=""
    # Nextcloud Storage Username (optional)
    USERNAME_STORAGE_NEXTCLOUD=""
    # Nextcloud Storage Password (optional)
    PASSWORD_STORAGE_NEXTCLOUD=""
    # Path inside Nextcloud for storage (optional)
    PATH_STORAGE_IN_NEXTCLOUD=""
    ```


## Data Structure

The project is structured into several folders:

```bash
├── assets
│   ├── data_samples        # Sample CSVs for tests/integration checks
│   └── events/videos       # Rendered example videos / demo assets
├── notebooks               # Exploratory and pipeline notebooks
└── src
   ├── fetcher_sportradar   # API fetch helpers (functions)
   ├── fetcher_kinexon      # API fetch helpers (functions)
   ├── pipelines            # Raw/normalized/synced/features/ml logic
   ├── hbl_etl_dagster      # Dagster assets/jobs/resources/defs
   └── apps                 # App entrypoints (e.g., simulator)
```

## Usage

### Dagster (current pipeline entrypoint)

The active Dagster definitions entrypoint is:

```bash
src/hbl_etl_dagster/defs.py
```

Run locally with:

```bash
dagster dev -m hbl_etl_dagster.defs
```

### Downloading Data

Raw season-level data (competition, season, teams, fixtures, Kinexon sessions) is executed via:

```bash
dagster job execute -m hbl_etl_dagster.defs -j season_raw_refresh_job
```

This job populates fixture partitions that are then used for fixture-level ingestion.

### Processing Data

Run the fixture pipeline (raw → normalized → synced → features → ML assets) for a fixture partition:

```bash
dagster job execute -m hbl_etl_dagster.defs -j fixture_raw_backfill_job --partition <fixture_id>
```

For interactive execution and monitoring in browser, run:

```bash
dagster dev -m hbl_etl_dagster.defs
```


### Direct Python modules (advanced)

Fetcher files under `src/fetcher_sportradar/` and `src/fetcher_kinexon/` are importable function modules and generally are not standalone CLI scripts.

### Feature Extraction

Feature extraction is part of `fixture_raw_backfill_job` via the Dagster assets in:

```bash
src/hbl_etl_dagster/assets_features/
```

### Training Machine Learning Model

Model training/inference assets are executed within the same fixture backfill job (assets in `src/hbl_etl_dagster/assets_ml/`).

If you want to run only the Python pipeline functions directly, use modules under:

```bash
src/pipelines/ml/
```

## Contributing

If you'd like to contribute to BieLeMetrics:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add my feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.
