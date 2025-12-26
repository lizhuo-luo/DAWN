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
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/gsm8k-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/gsm8k-ns0-${length}

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/gsm8k-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/gsm8k-ns0-${length}

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,outp_path=evals_results/g-dllm_test/gsm8k-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm_test/gsm8k-ns0-${length}

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=${factor},outp_path=evals_results/factor/gsm8k-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/gsm8k-ns0-${length}

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,outp_path=evals_results/klass/gsm8k-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/gsm8k-ns0-${length}

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/gsm8k-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/gsm8k-ns0-${length}

############################################### minerva_math evaluations ###############################################
task=minerva_math
length=256
block_length=32
num_fewshot=4
steps=256
model_path='/workplace/models/LLaDA-8B-Instruct'

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/math-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/math-ns0-${length}

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.9,outp_path=evals_results/parallel/math-ns0-${length}/results.jsonl \
# --output_path evals_results/parallel/math-ns0-${length}

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,outp_path=evals_results/g-dllm_test/math-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm_test/math-ns0-${length}

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=${factor},outp_path=evals_results/factor/math-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/math-ns0-${length}

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,outp_path=evals_results/klass/math-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/math-ns0-${length}

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/math-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/math-ns0-${length}

############################################### gsm8k evaluations ###############################################
task=humaneval
length=256
block_length=32
num_fewshot=0
steps=$((length / block_length))
model_path='/workplace/models/LLaDA-8B-Instruct'
factor=1

# baseline
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/humaneval-ns0-${length}/results.jsonl \
# --output_path evals_results/baseline/humaneval-ns0-${length} --log_samples

# parallel threshold
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} --limit ${limit} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.7 \
# --output_path evals_results/parallel/gsm8k-ns0-${length}

# g-dllm
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,g_dllm=True,outp_path=evals_results/g-dllm/humaneval-ns0-${length}/results.jsonl \
# --output_path evals_results/g-dllm/humaneval-ns0-${length} --log_samples

# factor
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,factor=${factor},outp_path=evals_results/factor/humaneval-ns0-${length}/results.jsonl \
# --output_path evals_results/factor/humaneval-ns0-${length} --log_samples

# klass
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True,klass=True,outp_path=evals_results/klass/humaneval-ns0-${length}/results.jsonl \
# --output_path evals_results/klass/humaneval-ns0-${length}

# local leap
# CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,local_leap=True,outp_path=evals_results/local_leap/humaneval-ns0-${length}/results.jsonl \
# --output_path evals_results/local_leap/humaneval-ns0-${length}

## NOTICE: use postprocess for humaneval
# python3 postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}

############################################### mbpp evaluations ###############################################
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
model_path='/workplace/models/LLaDA-8B-Instruct'

# baseline
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,outp_path=evals_results/baseline/mbpp-ns0-${length}/results.jsonl \
--output_path evals_results/baseline/mbpp-ns0-${length} --log_samples

## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}