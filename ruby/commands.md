## self stream

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m ruby.??? --bench_name classification_public --model_name "Qwen/Qwen2.5-7B-Instruct" --device "auto" --output_path ./output/prediction.csv --use_wandb


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m ruby.02_stream_magic_preface_reasoning --bench_name classification_public --model_name "Qwen/Qwen2.5-7B-Instruct" --device "auto" --output_path ./output/prediction.csv --use_wandb

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m ruby.06_stream_magic_preface_stored_structured_reasoning_clear_instructions --bench_name classification_public --model_name "Qwen/Qwen2.5-7B-Instruct" --device "auto" --output_path ./output/prediction.csv --use_wandb

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && python3 -m ruby.07_stream_medical_magic_preface_stored_structured_reasoning_clear_instructions --bench_name classification_public --model_name "Qwen/Qwen2.5-7B-Instruct" --device "auto" --output_path ./output/prediction.csv --use_wandb