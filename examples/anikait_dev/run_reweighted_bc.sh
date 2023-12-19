exp_num=0
export HF_DATASETS_CACHE="/iris/u/asap7772/.cache"
export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH

mix_ratios=(1.0) # fully preference data
use_gold_rews=(true)
wandb_project="12_16_reweighted_bc_4inner_largerbatch_norescale"
dryrun=false
debug=true
which_exp=${1:--1}
temperatures=(1.0 2.0 5.0 10.0 20.0 50.0)

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for mix_ratio in "${mix_ratios[@]}"; do
for use_gold_rew in "${use_gold_rews[@]}"; do
for temperature in "${temperatures[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    echo "mix_ratio: $mix_ratio"
    echo "use_gold_rew: $use_gold_rew"
    echo "temperature: $temperature"

    command="python /iris/u/asap7772/trl/examples/anikait_dev/reweighted_bc.py \
        --wandb_project $wandb_project \
        --run_name reweighted_bc_usegoldrew${use_gold_rew}_mixratio${mix_ratio}_temp${temperature} \
        --mixing_ratio ${mix_ratio} \
        --temperature ${temperature} \
        --use_score_scaling False \
        --use_score_norm False \
        --gradient_accumulation_steps 8 \
        --mini_batch_size 16 \
        --batch_size 128
    "

    if [[ $use_gold_rew = true ]]; then
        command+="--use_gold_reward_model "
    fi

    if [[ $which_exp -lt 0 ]]; then
        command+=" &"
    fi
    echo -e "$command\n"
    if [ $dryrun = false ]; then
        eval $command
        sleep 20
    fi
    exp_num=$((exp_num+1))

done
done
done    