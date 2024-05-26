
# Explanation Quality of Synthetic Data

This repository contains scripts for generating synthetic data and evaluating the explanation quality of this data across various metrics. The project is structured to run multiple objectives, each with its unique datasets and parameters.

## Prerequisites

Before you begin, ensure you have Python installed on your system. You can download Python from [python.org](https://www.python.org/downloads/).

## Setup

To set up your environment to run these scripts, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Generating Synthetic Data

Navigate to the `dgp/synthetic/` directory to run the data generation scripts:

```bash
cd dgp/synthetic/
```

Execute each script to generate synthetic data for each objective:

- **Objective 1:**
  ```bash
  python objective-1-DGP.py
  ```
- **Objective 2:**
  ```bash
  python objective-2-DGP.py
  ```
- **Objective 3:**
  ```bash
  python objective-3-DGP.py
  ```
- **Objective 4:**
  ```bash
  python objective-4-DGP.py
  ```

Each script will generate a CSV file and save it to a specified folder.

## Running Evaluation Scripts

Once the synthetic data is generated, you can evaluate the explanation quality by running the corresponding scripts in the root directory:

```bash
cd ../..
```

Run each evaluation script, ensuring you use the correct path and file for each objective:

- **Objective 1:**
  ```bash
  python objective-1-run.py
  ```
- **Objective 2:**
  ```bash
  python objective-2-run.py
  ```
- **Objective 3:**
  ```bash
  python objective-3-run.py
  ```
- **Objective 4:**
  ```bash
  python objective-4-run.py
  ```

Make sure to update the `BASE_PATH` and `test_data_path` in each script to point to the correct data files and directories.

## File Structure

- `dgp/synthetic/` - Contains scripts for generating synthetic data.
- `evaluation_metrics.py` - Provides metrics for evaluating explanation quality.
- `explanation.py` - Contains methods for generating LIME explanations.
- `model.py` - Includes the machine learning model definitions.
- `requirements.txt` - Lists all dependencies necessary to run the scripts.
