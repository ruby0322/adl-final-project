# Text-To-SQL Generation

This project involves building and running agents for tasks such as classification and SQL generation. Below is a description of the environment setup and the required packages.

## Environment

The project is designed to work in a Python environment with GPU support. Key environment settings include:

- **Python Version**: Python 3.8 or later.
- **CUDA**: Ensure CUDA is properly installed for GPU acceleration.
- **PyTorch**: Utilized for deep learning computations with GPU support.

## Required Packages

To run this project, the following Python packages are required:

1. **PyTorch**:
   - Provides deep learning capabilities, including model training and inference on GPUs.
   - Install via: `pip install torch`

2. **Transformers**:
   - Used for loading and fine-tuning large language models (e.g., `Qwen2.5-7B-Instruct`).
   - Install via: `pip install transformers`

3. **BitsAndBytes**:
   - Enables 8-bit quantization for efficient model inference.
   - Install via: `pip install bitsandbytes`

4. **Rank-BM25**:
   - Implements the BM25 algorithm for efficient text retrieval.
   - Install via: `pip install rank-bm25`

5. **Colorama**:
   - Used for colored terminal text outputs.
   - Install via: `pip install colorama`

6. **NumPy**:
   - Provides support for numerical computations and array manipulation.
   - Install via: `pip install numpy`

7. **Execution Pipeline**:
   - Custom module for orchestrating benchmarks and executing agents.
   - Ensure it is available in your Python environment.

8. **Utilities**:
   - Includes `RAG` for retrieval-augmented generation and `strip_all_lines` for cleaning text inputs.

9. **Re**:
   - Standard library module used for regular expression operations.

## Installation

To set up the environment, execute the following commands:

```bash
pip install torch transformers bitsandbytes rank-bm25 colorama numpy
```

## Execution

Simply run the commnad below in terminal

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m ruby.text_to_sql --bench_name sql_generation_public --model_name meta-llama/Llama-3.1-8B-Instruct --device "auto" --max_tokens 512 --RAG_topk 8 --RAG_embedding_model "BAAI/bge-large-en-v1.5" --output_path "./prediction.csv" --use_wandb --use_8bit
```

# Diagnosis classification

This project focuses on developing a Local Model Agent for text generation tasks, specifically classification and diagnosis. Below is an overview of the environment and the required packages.

## Environment

- **Python Version**: Python 3.8 or later is recommended.
- **CUDA**: The project supports GPU acceleration. Ensure CUDA is installed and configured for optimal performance.

## Required Packages

To set up the environment, the following packages are needed:

1. **re**: 
   - Standard Python library for regular expression operations.
   - Used for pattern matching and text processing.

2. **random**:
   - Standard Python library for random number generation.
   - Used for handling edge cases when predictions are ambiguous.

3. **colorama**:
   - Provides color formatting for terminal outputs.
   - Install via: `pip install colorama`

4. **torch**:
   - PyTorch library for deep learning computations.
   - Supports GPU acceleration.
   - Install via: `pip install torch`

5. **transformers**:
   - Hugging Face's library for pre-trained models and tokenizers.
   - Used to load and fine-tune models like Qwen2.5-7B-Instruct.
   - Install via: `pip install transformers`

6. **BitsAndBytes**:
   - Enables 8-bit quantization for efficient memory usage during model inference.
   - Install via: `pip install bitsandbytes`

7. **warnings**:
   - Standard Python library to handle and filter warnings.
   - Used here to suppress warnings for a cleaner output.

8. **transformers.logging**:
   - Part of the Hugging Face library to manage log verbosity.
   - Adjusted here to suppress unnecessary logs.

9. **os**:
   - Standard Python library for interacting with the operating system.
   - Used for path manipulations and configuration management.

10. **utils**:
    - Includes custom utility functions like `RAG` for retrieval-augmented generation and `strip_all_lines` for text processing.
    - Ensure the `utils` module is included in the project directory.

## Installation

To set up the required environment, run the following command:

```bash
pip install torch transformers bitsandbytes colorama
```
## Execution

Simply run the commnad below in terminal

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m ruby.07_stream_medical_magic_preface_stored_structured_reasoning_clear_instructions --bench_name classification_public --model_name "Qwen/Qwen2.5-7B-Instruct" --device "auto" --output_path "./prediction.csv" --use_wandb
```

`07_stream_medical_magic_preface_stored_structured_reasoning_clear_instructions.py` is the script that trains the best performing model of our team (acc 73.639%)