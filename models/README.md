# Model Weights Directory

This directory stores fine-tuned BERT and DistilBERT checkpoints for the fake news detection engine.

## Directory Structure
- `fine_tuned_bert/`: Default output destination for checkpoints trained via `python src/train.py`.
  - `config.json`
  - `model.safetensors`
  - `tokenizer_config.json`
  - `vocab.txt`

## 8GB Laptop Optimization Highlights
- **DistilBERT Base**: 66M parameters (~260MB footprint), runs with low memory overhead (<1.5GB total RAM).
- **Inference Acceleration**: Automatically offloads to Apple Silicon Metal Performance Shaders (`mps`) or NVIDIA CUDA (`cuda`), falling back to multi-core CPU.
- **Cache Evacuation**: Explicit `empty_cache()` calls are performed after inference batches to eliminate RAM accumulation.
