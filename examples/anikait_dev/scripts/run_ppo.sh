exp_num=0
export HF_DATASETS_CACHE="/iris/u/asap7772/.cache"
export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH

mix_ratios=(0.0)
wandb_project="12_27_ppo_rew_no_score_norm"
dryrun=false
debug=false
which_exp=${1:--1}

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for mix_ratio in "${mix_ratios[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    command="python /iris/u/asap7772/trl/examples/anikait_dev/ppo.py \
        --wandb_project $wandb_project \
        --run_name ppo_gold_rew_mixrat${mix_ratio} \
        --mixing_ratio ${mix_ratio} \
        --use_score_scaling True \
        --use_score_norm True \
    "

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
