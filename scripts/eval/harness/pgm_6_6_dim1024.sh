for task in lambada_openai arc_easy arc_challenge hellaswag mathqa piqa winogrande; do
  python mgm_eval_harness.py \
    --batch_size 32 \
    --tasks $task \
    --model mgm \
    --trust_remote_code \
    --model_args checkpoint_path=PATH-TO-YOUR-PGM-6-6-DIM1024-CHECKPOINT \
    --output_path ./harness_results/pgm_6_6_dim1024/$task.json
done

for task_config in "social_iqa 512" "openbookqa 256" "boolq 64" "race 16"; do
  task=$(echo $task_config | cut -d' ' -f1)
  batch_size=$(echo $task_config | cut -d' ' -f2)
  python mgm_eval_harness.py \
    --batch_size $batch_size \
    --tasks $task \
    --model mgm \
    --trust_remote_code \
    --model_args checkpoint_path=PATH-TO-YOUR-PGM-6-6-DIM1024-CHECKPOINT \
    --output_path ./harness_results/pgm_6_6_dim1024/$task.json
done
