for task in lambada_openai arc_easy arc_challenge hellaswag mathqa piqa winogrande; do
  python mgm_eval_harness.py \
    --batch_size 64 \
    --tasks $task \
    --model mgm \
    --trust_remote_code \
    --model_args checkpoint_path=PATH-TO-YOUR-MDLM-CHECKPOINT \
    --output_path ./harness_results/mdlm/$task.json
done

for task_config in "social_iqa 512" "openbookqa 256" "boolq 64" "race 32"; do
  task=$(echo $task_config | cut -d' ' -f1)
  batch_size=$(echo $task_config | cut -d' ' -f2)
  python mgm_eval_harness.py \
    --batch_size $batch_size \
    --tasks $task \
    --model mgm \
    --trust_remote_code \
    --model_args checkpoint_path=PATH-TO-YOUR-MDLM-CHECKPOINT \
    --output_path ./harness_results/mdlm/$task.json
done
