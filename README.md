# DA6401 Assignment 3
### DA24M006 Gaurav Kumar M.Tech DSAI
## Overview

The implementation uses a sequence-to-sequence model vanila (without attention) and also  with attention mechanism and supports various RNN cell types (RNN, GRU, LSTM). 
This code includes features for:
- Training models on different languages (default hindi)
- Hyperparameter optimization using Weights & Biases (W&B) sweeps
- Comprehensive evaluation with visualizations

Implements transliteration (converting text from one latin to native) using two approaches:
- Vanilla sequence-to-sequence model
- Sequence-to-sequence model with attention mechanism

## Dataset

Using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina) by Google Research.


## File  Structure

```
.
├── dakshina_dataset_v1.0/        # Dataset folder
│   └── hi/                       # Hindi language data
│   |   ├── lexicons/             # Train, dev, test files
│   |   │   ├── hi.translit.sampled.train.tsv
│   |   │   ├── hi.translit.sampled.dev.tsv
│   |   │   └── hi.translit.sampled.test.tsv
│   |   ├── native_script_wikipedia/
│   |   └── romanized/
|   └── pa/ ...
|
├── predictions_attention/        # Predictions from attention model
│   ├── correct_predictions.txt
│   └── incorrect_predictions.txt
├── predictions_vanila/           # Predictions from vanilla model
│   ├── correct_predictions.txt
│   └── incorrect_predictions.txt
├── seq2seqAttention/            # Attention-based model code
│   ├── config.py                # Configuration settings(model and training parameters)
│   ├── data_preprocess_load.py  # Data loading and preprocessing
│   ├── evaluation_attention.py  # Model evaluation and visualization on test dataset
│   ├── main_attention.py        # Main script to run train, sweep, evalution
│   ├── seq2seq_attention.py     # Sequence-to-sequence model implementation
│   ├── sweep_attention.py       # Hyperparameter tuning for attention based seq2seq
│   ├── train_attention.py       # Model training 
│   └── utils_fun.py            # Helper functions
├── seq2seqVanila/              # Vanilla model code
│   ├── config.py                # Configuration settings
│   ├── data_preprocess_load.py  # Data loading and preprocessing
│   ├── evaluation_vanila.py     # Model evaluation and visualization on test dataset
│   ├── main_vanila.py           # Main script to run everything
│   ├── seq2seq_vanila.py        # Sequence-to-sequence model with attention implementation
│   ├── sweep_vanila.py          # Hyperparameter tuning for vanila deq2seq
│   ├── train_vanila.py          # Model training
│   └── utils_fun.py            # Helper functions
├── seq2seqAttention.ipynb      # Notebook for attention experiments
└── seq2seqVanila.ipynb         # Notebook for vanilla experiments
```

## Setup

1. Clone the repo and navigate to the folder:
      ```bash
   git clone https://github.com/Gaurav-Kumar-da24/da6401_assignment3
   cd dakshina-transliteration
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install torch numpy pandas matplotlib seaborn tqdm wandb
   ```

4. Download the Dakshina dataset:
   ```bash
   # Download from https://github.com/google-research-datasets/dakshina
   # Extract to a directory named 'dakshina_dataset_v1.0' in the project root
   # or update the data_dir in config.py to point to your dataset location
   ```
5. Set up Weights & Biases (for experiment tracking):
   ```bash
   wandb login    #use your wandb api key
   ```

6. Update configuration:
   - Open `seq2seqVanila/config.py` or `seq2seqAttention/config.py`
   - Update the W&B settings:
     ```python
     wandb_entity = "da24m006-iit-madras"  # Change this to your W&B username
     wandb_project= "DL_A3_seq2seq"
     ```

## How to Use

### For Vanilla Model (Seq2seq without attention)

Navigate to the vanilla model folder:
```bash
cd seq2seqVanila
```

**Training:**
```bash
python main_vanila.py --mode train  # this will use hi (hindi as default language uses config.py file)
```

To train for a different language:

```bash
python main_vanila.py --mode train --language pa  # for punjabi 
```
**Hyperparameter Search:**
```bash
python main_vanila.py --mode sweep --count 50    # default epochs=20
```

**Evaluation:**
```bash
python main_vanila.py --mode evaluate --sweep_id <your-sweep-id>
```

### For Attention Model (Seq2seq with attention)

Navigate to the attention model folder:
```bash
cd seq2seqAttention
```

**Training:**
```bash
python main_attention.py --mode train    # this will use hi (hindi as default language)
```

To train for a different language:

```bash
python main_attention.py --mode train --language pa  # for punjabi 
```
**Hyperparameter Search:**
```bash
python main_attention.py --mode sweep --count 50
```

**Evaluation:** 

Evaluating Models on test dataset from sweep
**While evaluting model we must provide sweep_id**
```bash
python main_attention.py --mode evaluate --sweep_id <your-sweep-id>
```

### Custom Parameters

You can customize training with different parameters:
```bash
python main_vanila.py --language hi --mode train --embed_size 128 --hidden_size 512 --cell_type LSTM --batch_size 64 --learning_rate 0.001 --epochs 50
```

## Model Details

Both models use:
- Character-level tokenization
- Encoder-decoder architecture
- Support for RNN, GRU, and LSTM cells
- Teacher forcing during training
- Greedy decoding for inference

The attention model additionally includes:
- Attention mechanism to focus on relevant input characters
- Better handling of longer sequences

## Output

After evaluation, you'll find:
- Prediction files in `predictions_vanila/` or `predictions_attention/`
- Accuracy metrics
- Sample correct and incorrect predictions
- Error analysis visualizations
- All results are also logged to W&B

## Notebooks

The jupyter notebooks (`seq2seqAttention.ipynb` and `seq2seqVanila.ipynb`) contain experimental code and analysis.

## Notes

- Make sure the dataset path in config.py matches your folder structure
- The default language is Hindi (hi)
- Models and results are saved to W&B for easy tracking
- Check the predictions folder after evaluation to see model performance
