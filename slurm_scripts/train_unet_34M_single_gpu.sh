#!/bin/bash
#SBATCH --job-name=train_u-net
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=28-00:00:00
#SBATCH --output=/gpfs/scratch/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/an4477/slurm_logs/%x_%j.err

# === Create logs directory if it doesn't exist ===
mkdir -p /gpfs/scratch/an4477/slurm_logs

# === Load and activate conda environment ===
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Print environment info for reproducibility ===
echo "Python executable: $(which python)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
nvidia-smi

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# === Weights & Biases config ===
export WANDB_PROJECT=long_runs
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762

export TOKENIZERS_PARALLELISM=false

# === Change to project directory ===
cd /gpfs/data/brandeslab/Project/u-net/project/HuggingfaceTransformer/

# === Use a random master port for torch distributed ===
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    python_scripts/train_modernBERT.py \
    --run-name unet_34M_single_gpu \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train \
    --val-dataset-path /gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation \
    --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
    --output-dir /gpfs/data/brandeslab/model_checkpts \
    --hidden-size 512 \
    --num-hidden-layers 8 \
    --num-attention-heads 8 \
    --intermediate-size 2048 \
    --local-attention 256 \
    --max-steps 3_000_000 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 128 \
    --per_device_eval_batch_size 4 \
    --base_batch_size 8 \
    --learning_rate 1e-3 \
    --vep_eval_steps 25_000 \
    --dataloader_num_workers 6 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --eval_strategy "no" \
    --save_steps 25_000 \
    --dynamic-batching


