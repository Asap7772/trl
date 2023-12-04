# export WANDB_MODE="dryrun"
export HF_DATASETS_CACHE="/iris/u/asap7772/.cache"
export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH
python /iris/u/asap7772/trl/examples/anikait_dev/reweighted_bc.py --wandb_project reweighted_bc --run_name reweighted_bc_gold_rew --use_gold_reward_model