# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model_path='/workplace/models/LLaDA-8B-Instruct'
factor=1

# baseline
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/baseline/gsm8k-ns0-${length}

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.7 \
# --output_path evals_results/parallel/gsm8k-ns0-${length}

# g-dllm
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,outp_path=evals_results/g-dllm/gsm8k-ns0-${length}/results.jsonl \
--output_path evals_results/g-dllm/gsm8k-ns0-${length}

# factor
CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=${factor},outp_path=evals_results/factor/gsm8k-ns0-${length}/results.jsonl \
--output_path evals_results/factor/gsm8k-ns0-${length}

############################################### gsm8k evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
model_path='/workplace/models/LLaDA-8B-Instruct'
factor=1

# baseline
# CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True \
# --output_path evals_results/baseline/gsm8k-ns0-${length}

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.7 \
# --output_path evals_results/parallel/gsm8k-ns0-${length}

# g-dllm
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,outp_path=evals_results/g-dllm/humaneval-ns0-${length}/results.jsonl \
--output_path evals_results/g-dllm/humaneval-ns0-${length}

# factor
CUDA_VISIBLE_DEVICES=1 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=${factor},outp_path=evals_results/factor/humaneval-ns0-${length}/results.jsonl \
--output_path evals_results/factor/humaneval-ns0-${length}
