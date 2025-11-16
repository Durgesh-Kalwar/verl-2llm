set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

nproc_per_node=2
ROLLOUT_TP_SIZE=2
PROJECT_NAME=gsm8k-sft

# BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct
BASE_MODEL=Qwen/Qwen2.5-1.5B
# BASE_MODEL=EleutherAI/pythia-1.4b-deduped-v0

# EXPERIMENT_NAME=gsm8k_llama3.2_3b_sft
EXPERIMENT_NAME=gsm8k_qwen2.5_1.5b_sft_lora_sp2
# EXPERIMENT_NAME=gsm8k_pythia_1.4b_sft

save_path=/scratch/dkalwar/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}
# Shift the arguments so $@ refers to the rest
shift 2
export LD_LIBRARY_PATH=/home/dkalwar/anaconda3/envs/2llm/lib:$LD_LIBRARY_PATH
# export VLLM_USE_V1=0

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/scratch/dkalwar/verl-2llm/data/gsm8k/train.parquet \
    data.val_files=/scratch/dkalwar/verl-2llm/data/gsm8k/test.parquet \
    data.train_batch_size=128 \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-6 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=64 \
    data.max_length=1024 \
    model.partial_pretrain=$BASE_MODEL \
    model.trust_remote_code=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=50 \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    +trainer.val_before_train=true \
    +trainer.val_only=false \
    +trainer.log_val_generations=0 \
    rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=$ROLLOUT_TP_SIZE \
    rollout.gpu_memory_utilization=0.8 \
    use_remove_padding=true
