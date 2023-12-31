# invert 30 layers of vicuna 7B model by one prompt
# python invert.py --base-model-name lmsys/vicuna-7b-v1.5 \
# --dataset-path ../skytrax-reviews-dataset/data/airline.csv \
# --dataset-type github --dataset-len 1 \
# --num-invert-layers 30 --output-dir results/7b-airline-30layer-results

python invert.py --base-model-name huggyllama/llama-65B \
--dataset-path ../skytrax-reviews-dataset/data/airline.csv \
--dataset-type github --dataset-len 100 \
--num-invert-layers 60 --output-dir results/65B-airline-60layer-results