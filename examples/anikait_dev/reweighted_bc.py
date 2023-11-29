from datasets import load_dataset
from trl import PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, ReweightedBCConfig, ReweightedBCTrainer
from transformers import pipeline
import torch
from absl import flags, app
import os
import accelerate
import datetime
import numpy as np
import tempfile
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'reweighted_bc', 'the wandb project name')
flags.DEFINE_string('run_name', 'reweighted_bc', 'the wandb run name')
flags.DEFINE_string('output_dir', None, 'the output directory')
flags.DEFINE_string('dataset_path', "tatsu-lab/alpaca_farm", 'the path to the dataset')
flags.DEFINE_string('tokenizer_type', "EleutherAI/pythia-1.4b", 'the model name')
flags.DEFINE_string('pretrained_dir', "/iris/u/asap7772/trl/output_checkpoints/checkpoint-7500", 'the path to the pretrained model')
flags.DEFINE_string('reward_model', "/iris/u/asap7772/conservative_reward_model/model_checkpoints_rewpref/rewpref_fixpad_labnoise_14m_1127/EleutherAI_pythia-14m_relabeled_alpacafarm_pythiasft_20K_preference_data_19000_0.0_1e-05_0.0/20231127-120834/epoch_5/", 'the path to the reward model')
flags.DEFINE_float('learning_rate', 1.0e-6, 'the learning rate')
flags.DEFINE_float('cosine_annealing_lr_eta_min', 1.0e-7, 'the cosine annealing eta min')
flags.DEFINE_integer('num_train_epochs', 4, 'the number of training epochs')
flags.DEFINE_integer('num_rollouts', 256, 'the number of rollouts')
flags.DEFINE_float('clip_range', 0.2, 'the clip range')
flags.DEFINE_float('gae_lambda', 0.95, 'the GAE lambda')
flags.DEFINE_integer('batch_size', 256, 'the batch size')
flags.DEFINE_integer('max_gen_batch_size', 16, 'the max generation batch size')
flags.DEFINE_integer('mini_batch_size', 32, 'the chunk size')
flags.DEFINE_integer('seed', 42, 'the random seed')
flags.DEFINE_string('filter_or_reweight', 'reweight', 'the type of reweighting to use')
# flags for reweighting
flags.DEFINE_float('temperature', 1.0, 'the temperature for reweighting')
flags.DEFINE_string('weighting_type', 'softmax', 'the type of reweighting to use')
flags.DEFINE_boolean('clip_weighting', True, 'whether to use score scaling')
flags.DEFINE_float('clip_min', -10.0, 'the minimum value for score scaling')
flags.DEFINE_float('clip_max', 10.0, 'the maximum value for score scaling')
# flags for filtering
flags.DEFINE_string('filter_type', 'threshold', 'the type of filtering to use')
flags.DEFINE_float('filter_threshold', 0.0, 'the filter threshold')
flags.DEFINE_integer('filter_topk', 3, 'the filter threshold')

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'

def main(_):
    dataset = load_dataset(FLAGS.dataset_path, split="unlabeled")
    eval_dataset = load_dataset(FLAGS.dataset_path, split="val")

    output_dir = FLAGS.output_dir or f"/iris/u/asap7772/trl/{FLAGS.wandb_project}/{FLAGS.run_name}"
    model_name = os.path.basename(output_dir)

    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    wandb_output_dir = tempfile.mkdtemp()
    config = ReweightedBCConfig(
        model_name=FLAGS.pretrained_dir,
        learning_rate=FLAGS.learning_rate,
        reward_model=FLAGS.reward_model,
        lam=FLAGS.gae_lambda,
        cliprange=FLAGS.clip_range,
        cliprange_value=FLAGS.clip_range,
        batch_size=FLAGS.batch_size,
        mini_batch_size=FLAGS.mini_batch_size,
        ppo_epochs=FLAGS.num_train_epochs,
        tracker_project_name=FLAGS.wandb_project,
        use_score_scaling=True,
        use_score_norm=True,
        filter_or_reweight=FLAGS.filter_or_reweight,
        temperature=FLAGS.temperature,
        weighting_type=FLAGS.weighting_type,
        clip_weighting=FLAGS.clip_weighting,
        clip_weighting_value_min=FLAGS.clip_min,
        clip_weighting_value_max=FLAGS.clip_max,
        filter_type=FLAGS.filter_type,
        filter_threshold=FLAGS.filter_threshold,
        filter_topk=FLAGS.filter_topk,
        project_kwargs={
            'project_dir': output_dir,
        },
        tracker_kwargs={
            "wandb": {
                "name": FLAGS.run_name, 
                "id": unique_str,
                "dir": wandb_output_dir,
            }
        },
        log_with='wandb',
        seed=FLAGS.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_type)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    eos = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(FLAGS.pretrained_dir)
    policy.resize_token_embeddings(len(tokenizer))
    model = AutoModelForCausalLMWithValueHead(policy)

    def formatting_prompts_func(example):
        inst, inp = example['instruction'], example['input']
        if inp:
            query = f"{PROMPT_TOKEN}{inst}\n{inp}{eos}{ASSISTANT_TOKEN}"
        else:
            query = f"{PROMPT_TOKEN}{inst}{eos}{ASSISTANT_TOKEN}"
        example['query'] = query
        return example

    dataset = dataset.map(formatting_prompts_func, batched=False)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=False)

    print('Sample Train prompt:', dataset[0]['query'])
    print('Sample Eval prompt:', eval_dataset[0]['query'])

    rbc_trainer = ReweightedBCTrainer(
        model=model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
    )

    generation_kwargs = {
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 256, # specify how many tokens you want to generate at most
        "temperature": 1.0, # control the temperature of the softmax, 1.0 means no change, lower means more greedy, higher means more diverse
        "use_cache": True, # whether or not the model should use the past key/values attentions (if the model supports it)
    }
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(FLAGS.reward_model)
    reward_tokenizer = AutoTokenizer.from_pretrained(FLAGS.reward_model)
    reward_model = rbc_trainer.accelerator.prepare_model(reward_model)
    reward_model.eval()

    print("Loaded reward model")

    def get_pred_reward(text, max_len=512):
        with torch.no_grad():
            encoded_input = reward_tokenizer(text, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            encoded_input = accelerate.utils.send_to_device(encoded_input, rbc_trainer.accelerator.device)
            output = reward_model(**encoded_input)
            logits = output.logits.squeeze()
        return logits
    
    def save_model(checkpoint_dir, epoch_num, add_prefix=True):
        if add_prefix:
            checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        if rbc_trainer.accelerator.is_main_process:
            rbc_trainer.accelerator.unwrap_model(model).save_pretrained(
                checkpoint_dir,
                save_function=rbc_trainer.accelerator.save,
                is_main_process=rbc_trainer.accelerator.is_main_process,
                state_dict=rbc_trainer.accelerator.get_state_dict(model),
            )
            if rbc_trainer.accelerator.is_main_process:
                tokenizer.save_pretrained(checkpoint_dir)
            rbc_trainer.accelerator.print(f"Checkpointing Epoch {epoch_num} -> {checkpoint_dir}")
 

    last_epoch = -1
    # TODO (anikait): setup batch mixing with preference dataset
    for epoch, batch in tqdm(enumerate(rbc_trainer.dataloader)):
        #### Construct query tensors
        query_tensors = tokenizer(batch["query"], padding=True, truncation=True, max_length=128, return_tensors='pt')
        query_tensors = accelerate.utils.send_to_device(query_tensors, rbc_trainer.accelerator.device)

        #### Get generations from SFTModel (including prompt)
        if query_tensors.input_ids.shape[0] > FLAGS.max_gen_batch_size:
            generation_tokens = []
            for i in tqdm(range(0, query_tensors.input_ids.shape[0], FLAGS.max_gen_batch_size), desc=f"Generating for epoch {epoch}"):
                generation_tokens.append(rbc_trainer.accelerator.unwrap_model(rbc_trainer.model).generate(**query_tensors[i:i+FLAGS.max_gen_batch_size], **generation_kwargs))
                torch.cuda.empty_cache()
            generation_tokens = torch.cat(generation_tokens, dim=0)
        else:
            generation_tokens = rbc_trainer.accelerator.unwrap_model(rbc_trainer.model).generate(**query_tensors, **generation_kwargs)
        texts = tokenizer.batch_decode(generation_tokens, skip_special_tokens=True)

        #### Update batch with response
        batch["response"] = [x.split(ASSISTANT_TOKEN)[-1] for x in texts]
        response_tensors = tokenizer(batch["response"], padding=True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
        response_tensors = [response_tensors[i] for i in range(0, len(response_tensors))]

        #### Compute reward score
        rewards = get_pred_reward(texts)
        rewards = [rewards[i] for i in range(0, len(rewards))]
        
        ### Reprocess query tensors
        query_tensors = query_tensors.input_ids
        query_tensors = [query_tensors[i] for i in range(0, len(query_tensors))]

        #### Run PPO step
        stats = rbc_trainer.step(query_tensors, response_tensors, rewards)
        rbc_trainer.log_stats(stats, batch, rewards)
        
        if epoch > 0 and epoch > last_epoch:
            save_model(model_name + f"_epoch_{epoch}", epoch)
            last_epoch = epoch

    #### Save model
    save_model(model_name, -1)

if __name__ == "__main__":
    app.run(main)