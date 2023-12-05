exp_num=0
export HF_DATASETS_CACHE="/iris/u/asap7772/.cache"
export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH

mix_ratios=(0.5 0.0)
use_gold_rews=(true false)
wandb_project="reweighted_bc"
dryrun=false
debug=false
which_exp=${1:--1}

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for mix_ratio in "${mix_ratios[@]}"; do
for use_gold_rew in "${use_gold_rews[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    echo "mix_ratio: $mix_ratio"
    echo "use_gold_rew: $use_gold_rew"

    command="python /iris/u/asap7772/trl/examples/anikait_dev/reweighted_bc.py \
        --wandb_project $wandb_project \
        --run_name reweighted_bc_usegoldrew${use_gold_rew}_mixratio${mix_ratio} \
        --mixing_ratio ${mix_ratio} \
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