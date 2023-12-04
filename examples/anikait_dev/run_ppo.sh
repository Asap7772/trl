# export WANDB_MODE="dryrun"
export PYTHONPATH=/iris/u/asap7772/trl:$PYTHONPATH
python /iris/u/asap7772/trl/examples/anikait_dev/ppo.py --wandb_project ppo_rew_padfix_rew --run_name ppo_gold_rew --use_gold_reward_model