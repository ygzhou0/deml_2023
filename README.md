# deml_attack
## Quick Run
For a quick run, try running `python invert.py --base_model_name lmsys/vicuna-7b-v1.5`. This will run decentralized inversion attack on vicuna 7B model. Please follow `instruction.txt` to set up a python environment.

## settings
### model
Use `--base-model-name` to set model name or model path in huggingface format, such as `lmsys/vicuna-7b-v1.5`, `huggyllama/llama-65b`.
Use `--lora-model-name` to add a huggingface lora adapter, which aligns to the base model.

### dataset
Use `--dataset-path` and `--dataset-type` to specify dataset. Dataset type can be github dataset, local json file, or huggingface datasets.

### hyper-parameters
Use `--lr`, `--epoch`, `--alpha`, `--perplexity` to set the corresponding parameters.

## metrics
Please refer to `metric.py` for evaluation metrics.