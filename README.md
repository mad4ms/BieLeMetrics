
# BieLeMetrics (Bielefeld Lemgo Metrics)

BieLeMetrics is a Python-based project aimed at processing handball data from Kinexon and Sportradar sources. The project includes downloading, synchronizing, and extracting features from event data to train machine learning models, such as an expected goal model, using the MLJAR platform.

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

- Python 3.7+
- [Conda](https://docs.conda.io/en/latest/) or [virtualenv](https://virtualenv.pypa.io/en/latest/)
- Required Python packages: See `requirements.txt`
- Git submodules (to initialize external libraries)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BieLeMetrics.git
   cd BieLeMetrics
   ```

2. Initialize submodules:
   ```bash
   git submodule init
   git submodule update
   ```

3. Create and activate the Python environment:
   ```bash
   conda create -n bielemetrics python=3.12
   conda activate bielemetrics
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Configure environment variables:
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
├── data
│   ├── events              # Event data from Sportradar and Kinexon
│   ├── ml_stuff            # Machine learning-related files and outputs
│   ├── processed           # Processed data ready for feature extraction
│   └── raw                 # Raw data from sources
└── src
    ├── helper_download      # Scripts for downloading data
    ├── helper_ml            # Machine learning helper functions
    ├── helper_preprocessing # Preprocessing scripts for feature extraction
    ├── utils                # Utility scripts
    └── libs_external        # External libraries used
```

## Usage

### Downloading Data

You can download game data for specific game IDs using:

```bash
python src/download_game_by_id.py <game_id>
```

To download games for an entire game day in parallel:

```bash
python src/download_gamedays.py
```

### Processing Data

After downloading, process the data by synchronizing and extracting features using:

```bash
python src/process_game.py <sportradar_path> <kinexon_path>
```

Or to process multiple game days in parallel:

```bash
python src/process_gamedays.py
```

### Feature Extraction

Processed game data will be saved in the `data/processed/` directory, where features are extracted into CSV files for training in MLJAR.

### Training Machine Learning Model

Once the feature extraction is completed, the resulting CSV files can be fed into MLJAR to train an expected goals model.

## Contributing

If you'd like to contribute to BieLeMetrics:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add my feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License.
