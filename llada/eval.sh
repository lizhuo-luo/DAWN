# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

model_path='/workplace/models/LLaDA/LLaDA-8B-Instruct'
model_name='LLaDA-8B-Instruct'

############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length}

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length}

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm/${task}-ns0-${length}

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=1,outp_path=evals_results/factor/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/${task}-ns0-${length}

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit 1 \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,conf_threshold=0.6,kl_threshold=0.015,outp_path=evals_results/klass/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/${task}-ns0-${length}

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/${task}-ns0-${length}

############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=256

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length}

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length}

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm/${task}-ns0-${length}

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=1,outp_path=evals_results/factor/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/${task}-ns0-${length}

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,conf_threshold=0.6,kl_threshold=0.01,outp_path=evals_results/klass/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/${task}-ns0-${length}

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/${task}-ns0-${length}

############################################### gsm8k evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length} --log_samples

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length} --log_samples

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm/${task}-ns0-${length} --log_samples

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=1,outp_path=evals_results/factor/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/${task}-ns0-${length} --log_samples

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,conf_threshold=0.9,kl_threshold=0.01,outp_path=evals_results/klass/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/${task}-ns0-${length} --log_samples

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/${task}-ns0-${length} --log_samples

## NOTICE: use postprocess for humaneval
# python3 postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}

############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/${task}-ns0-${length} --log_samples

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/${task}-ns0-${length} --log_samples

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,threshold_c=0.7,outp_path=evals_results/g-dllm/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm/${task}-ns0-${length} --log_samples

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=1,outp_path=evals_results/factor/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/${task}-ns0-${length} --log_samples

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,conf_threshold=0.7,kl_threshold=0.01,outp_path=evals_results/klass/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/${task}-ns0-${length} --log_samples

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/${task}-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/${task}-ns0-${length} --log_samples

## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}