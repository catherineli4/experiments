# Alpaca Dataset Log Likelihood Quartile Splitter

This script takes an Alpaca dataset, computes log likelihood scores using the OLMo2 1B HF model, and splits the dataset into 4 quartiles based on those scores.

## Features

- Loads OLMo2 1B model from HuggingFace
- Computes log likelihood scores for each sample in the Alpaca dataset
- Splits dataset into 4 quartiles:
  - `top_25`: Top 25% (highest log likelihood scores)
  - `q25_50`: 25-50% quartile
  - `q50_75`: 50-75% quartile  
  - `bottom_25`: Bottom 25% (lowest log likelihood scores)
- Saves each quartile to separate JSON files
- Provides detailed statistics for each quartile

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```bash
python alpaca_loglikelihood_split.py
```

This will:
- Use the default Alpaca dataset path
- Load the OLMo2 1B model from HuggingFace
- Process all samples and save quartiles to `alpaca_quartiles/`

### Advanced Usage

```bash
python alpaca_loglikelihood_split.py \
    --dataset_path path/to/your/alpaca_dataset.json \
    --model_name allenai/OLMo-2-0425-1B \
    --output_dir my_quartiles \
    --max_samples 1000
```

### Arguments

- `--dataset_path`: Path to the Alpaca dataset JSON file (default: uses the provided Alpaca dataset)
- `--model_name`: HuggingFace model name for OLMo2 1B (default: "allenai/OLMo-2-0425-1B")
- `--output_dir`: Output directory for quartile files (default: "alpaca_quartiles")
- `--max_samples`: Maximum number of samples to process (for testing, default: None)

## Output

The script creates the following files in the output directory:

- `top_25.json`: Samples with highest log likelihood scores
- `q25_50.json`: Samples in the 25-50% range
- `q50_75.json`: Samples in the 50-75% range  
- `bottom_25.json`: Samples with lowest log likelihood scores

Each sample in the output files includes:
- Original Alpaca fields (`instruction`, `input`, `output`)
- `log_likelihood`: The computed log likelihood score
- `rank`: The rank of the sample (1 = highest log likelihood)

## Example Output

```
Loading model: allenai/OLMo-2-0425-1B
Loading dataset from: OLMo/inference/compression/dependencies/AutoGPTQ/examples/quantization/dataset/alpaca_data_cleaned.json
Loaded 52002 samples
Computing log likelihood scores...
Processing samples: 100%|██████████| 52002/52002 [05:23<00:00, 161.23it/s]

==================================================
QUARTILE STATISTICS
==================================================

TOP_25:
  Number of samples: 13000
  Mean log likelihood: -245.67
  Std log likelihood: 45.23
  Min log likelihood: -320.45
  Max log likelihood: -180.12

Q25_50:
  Number of samples: 13001
  Mean log likelihood: -356.78
  Std log likelihood: 52.34
  Min log likelihood: -420.67
  Max log likelihood: -320.46

Q50_75:
  Number of samples: 13000
  Mean log likelihood: -478.90
  Std log likelihood: 61.45
  Min log likelihood: -550.23
  Max log likelihood: -420.68

BOTTOM_25:
  Number of samples: 13001
  Mean log likelihood: -612.34
  Std log likelihood: 78.90
  Min log likelihood: -750.12
  Max log likelihood: -550.24

Saved 13000 samples to alpaca_quartiles/top_25.json
  Log likelihood range: -180.12 to -320.45
Saved 13001 samples to alpaca_quartiles/q25_50.json
  Log likelihood range: -320.46 to -420.67
Saved 13000 samples to alpaca_quartiles/q50_75.json
  Log likelihood range: -420.68 to -550.23
Saved 13001 samples to alpaca_quartiles/bottom_25.json
  Log likelihood range: -550.24 to -750.12

Processing complete! Quartiles saved to alpaca_quartiles/
```

## Notes

- The script uses the OLMo2 1B model which requires sufficient GPU memory
- Log likelihood computation can be time-consuming for large datasets
- Use `--max_samples` for testing with a smaller subset
- The script handles errors gracefully and assigns very low log likelihood scores to failed samples 