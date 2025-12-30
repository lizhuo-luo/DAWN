# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model="/workplace/models/Dream/Dream-v0-Instruct-7B"

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/baseline/${task}-ns0-${length}


# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/parallel/${task}-ns0-${length}

# g-dllm
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,threshold_e=0.1,threshold_d=0.9,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results/g-dllm/${task}-ns0-${length}

############################################### minerva_math evaluations ###############################################
task=math
length=256
block_length=32
num_fewshot=4
steps=$((length / block_length))
model="/workplace/models/Dream/Dream-v0-Instruct-7B"

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/baseline/${task}-ns0-${length}


# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/parallel/${task}-ns0-${length}

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,threshold_e=0.1,threshold_d=0.9,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/g-dllm/${task}-ns0-${length}
    
############################################### humaneval evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))
model="/workplace/models/Dream/Dream-v0-Instruct-7B"

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/baseline/${task}-ns0-${length} --log_samples


# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/parallel/${task}-ns0-${length} --log_samples

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,threshold_e=0.1,threshold_d=0.9,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/g-dllm/${task}-ns0-${length} --log_samples

############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=$((length / block_length))
model="/workplace/models/Dream/Dream-v0-Instruct-7B"

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},add_bos_token=true,alg=entropy,show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/baseline/${task}-ns0-${length} --log_samples


# parallel
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},block_length=${block_length},add_bos_token=true,alg=confidence_threshold,threshold=0.9,show_speed=True,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/parallel/${task}-ns0-${length} --log_samples

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py --model dream \
#     --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${length},block_length=${block_length},add_bos_token=true,alg=g-dllm,show_speed=True,threshold_e=0.1,threshold_d=0.9,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
#     --tasks ${task} \
#     --num_fewshot ${num_fewshot} \
#     --batch_size 1 \
#     --output_path evals_results/g-dllm/${task}-ns0-${length} --log_samples