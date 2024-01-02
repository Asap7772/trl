exp_num=0
export HF_DATASETS_CACHE="/iris/u/asap7772/.cache"
export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH

mix_ratios=(0.0)
wandb_project="12_31_ppo_ablation"
dryrun=false
debug=false
which_exp=${1:--1}
target_kls=(2.0 4.0 6.0)
vf_coefs=(0.1 0.5)

if [[ $debug = true ]]; then
    echo "Running in debug mode"
    export WANDB_MODE="dryrun"
fi

for mix_ratio in "${mix_ratios[@]}"; do
for target_kl in "${target_kls[@]}"; do
for vf_coef in "${vf_coefs[@]}"; do
    if [[ $exp_num != $which_exp && $which_exp -ge 0 ]]; then
        exp_num=$((exp_num+1))
        continue
    fi

    command="python /iris/u/asap7772/trl/examples/anikait_dev/ppo.py \
        --wandb_project $wandb_project \
        --run_name ppo_mixrat${mix_ratio}_kltarg${target_kl} \
        --mixing_ratio ${mix_ratio} \
        --inner_iteration_steps 1 \
        --batch_size 16 \
        --max_gen_batch_size 8 \
        --mini_batch_size 8 \
        --target_kl ${target_kl} \
        --vf_coef ${vf_coef} \
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
done
done