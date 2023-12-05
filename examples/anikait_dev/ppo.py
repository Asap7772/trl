import os
os.environ["WANDB__SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from trl import PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from alpaca_farm.models.reward_model import RewardModel, RewardConfig
from transformers import pipeline
import torch
from absl import flags, app
import os
import accelerate
import datetime
import numpy as np
import tempfile
from tqdm import tqdm
import re

FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'ppo_rew_padfix_rew', 'the wandb project name')
flags.DEFINE_string('run_name', 'ppo_rew_labnoise0.0_rewnorm', 'the wandb run name')
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
flags.DEFINE_integer('mini_batch_size', 8, 'the chunk size')
flags.DEFINE_integer('seed', 42, 'the random seed')
# score manipulation
flags.DEFINE_bool('use_score_scaling', False, 'whether to use score scaling')
flags.DEFINE_bool('use_score_norm', False, 'whether to use score normalization')
# flags for preference dataset
flags.DEFINE_string('preference_dataset_path', '/iris/u/asap7772/conservative_reward_model/data_trl/relabeled_alpacafarm_pythiasft_20K_preference_data', 'the path to the preference dataset')
flags.DEFINE_integer('preference_num_samples', 19000, 'the number of samples to use from the preference dataset')
flags.DEFINE_bool("batched", True, "Whether to use batched processing")
flags.DEFINE_integer("num_proc", 32, "Number of processes to use")
flags.DEFINE_float('mixing_ratio', 0.0, 'the mixing ratio for preference dataset')
# flags to setup gold reward model.
flags.DEFINE_bool('use_gold_reward_model', False, 'whether to use gold reward model')
flags.DEFINE_string('gold_reward_model_path', '/iris/u/asap7772/downloaded_models/reward-model-human', 'the path to the reward model')
flags.DEFINE_integer('gold_shard_size', 8, 'the number of shards to use for gold reward model')
flags.DEFINE_string('sft10k_path', '/iris/u/asap7772/downloaded_models/sft10k', 'the path to the sft10k model')
flags.DEFINE_bool('flash_attn', False, 'whether to use flash attention')

def get_dataset(path, num_samples=-1, return_test_data=True, num_samples_test=1000):
    assert os.path.exists(path)
    folders = os.listdir(path)
    regex = r"^\d+-\d+$"
    folders = [x for x in folders if re.search(regex, x)]
    folders.sort(key=lambda x: int(x.split("-")[0]))
    total_samples = int(folders[-1].split("-")[-1])

    assert 0 < num_samples <= total_samples - num_samples_test, f"num_samples {num_samples} must be between 0 and {total_samples} - {num_samples_test}"
    assert 0 < num_samples_test <= total_samples, f"num_samples_test {num_samples_test} must be between 0 and {total_samples}"
    
    num_samples_train = num_samples if num_samples > 0 else total_samples - num_samples_test
    test_folders = [x for x in folders if int(x.split("-")[0]) >= num_samples_train]
    folders = [x for x in folders if int(x.split("-")[0]) < num_samples_train]
    
    datasets = [load_from_disk(os.path.join(path, x)) for x in folders]
    full_data =  concatenate_datasets(datasets)
    
    if num_samples > 0:
        full_data = full_data.select(range(num_samples))
        
    if return_test_data:
        test_datasets = [load_from_disk(os.path.join(path, x)) for x in test_folders]
        test_data = concatenate_datasets(test_datasets)
        if num_samples_test > 0:
            test_data = test_data.select(range(num_samples_test))
        return full_data, test_data
    
    return full_data
    

def construct_dataset(
    path,
    num_samples=-1,
    concatenate_prompt=False,
    num_samples_test=1000,
):
    data, test_data = get_dataset(path, num_samples=num_samples, return_test_data=True, num_samples_test=num_samples_test)

    if concatenate_prompt:
        def map_fn(d):
            for k in ["y_ref", "y_w", "y_l"]:
                d[k] = d["prompt"] + d[k]
            return d
        
        data = data.map(
            map_fn,
            num_proc=FLAGS.num_proc,
        )

    dataset_name = os.path.basename(path).split(".")[0]

    ds = DatasetDict(
        {
            "train": data,
            "test": test_data,
        }
    ) 
    return dataset_name, ds


PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'

def main(_):
    dataset = load_dataset(FLAGS.dataset_path, split="unlabeled")
    eval_dataset = load_dataset(FLAGS.dataset_path, split="val")

    output_dir = FLAGS.output_dir or f"/iris/u/asap7772/trl/{FLAGS.wandb_project}/{FLAGS.run_name}"
    model_name = f"{FLAGS.wandb_project}_{FLAGS.run_name}"
    
    print('Output dir:', output_dir)
    print('Model name:', model_name)

    batch_size_pref_data = int(FLAGS.batch_size * FLAGS.mixing_ratio)
    batch_size_online_data = FLAGS.batch_size - batch_size_pref_data
    
    if FLAGS.preference_dataset_path == "tatsu-lab/alpaca_farm":
        pref_dataset = load_dataset(FLAGS.preference_dataset_path, split="sft")
        eval_pref_dataset = load_dataset(FLAGS.preference_dataset_path, split="preference")
    else:
        pref_dataset_name, pref_dataset = construct_dataset(
            path=FLAGS.preference_dataset_path,
            num_samples=FLAGS.preference_num_samples,
            concatenate_prompt=False,
        )
        print('Loaded dataset', pref_dataset_name)
        pref_dataset, eval_pref_dataset = pref_dataset['train'], pref_dataset['test']
    
    def process_dataset(batch):
        new_batch = {}
        new_batch['query'] = batch['prompt'] + batch['prompt']
        new_batch['text'] =  batch['y_w'] + batch['y_l']
        new_batch['response'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w'] + batch['y_l']]
        return new_batch

    remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']

    pref_dataset = pref_dataset.map(
        process_dataset,
        batched=FLAGS.batched,
        num_proc=FLAGS.num_proc,
        remove_columns=remove_columns,
    )
    
    eval_pref_dataset = eval_pref_dataset.map(
        process_dataset,
        batched=FLAGS.batched,
        num_proc=FLAGS.num_proc,
        remove_columns=remove_columns,
    )

    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    wandb_output_dir = tempfile.mkdtemp()
    config = PPOConfig(
        model_name=FLAGS.pretrained_dir,
        learning_rate=FLAGS.learning_rate,
        reward_model=FLAGS.reward_model,
        lam=FLAGS.gae_lambda,
        cliprange=FLAGS.clip_range,
        cliprange_value=FLAGS.clip_range,
        batch_size=FLAGS.batch_size,
        dataloader_batch_size=max(batch_size_online_data, 1),
        mini_batch_size=FLAGS.mini_batch_size,
        ppo_epochs=FLAGS.num_train_epochs,
        tracker_project_name=FLAGS.wandb_project,
        use_score_scaling=FLAGS.use_score_scaling,
        use_score_norm=FLAGS.use_score_norm,
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

    trainer = PPOTrainer(
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
    if not FLAGS.use_gold_reward_model:
        reward_model = trainer.accelerator.prepare_model(reward_model)
    reward_model.eval()

    print("Loaded reward model")

    def get_pred_reward(text, max_len=512):
        with torch.no_grad():
            encoded_input = reward_tokenizer(text, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            encoded_input = accelerate.utils.send_to_device(encoded_input, trainer.accelerator.device)
            output = reward_model(**encoded_input)
            logits = output.logits.squeeze()
        return logits

    gold_tokenizer = AutoTokenizer.from_pretrained(FLAGS.gold_reward_model_path)

    gold_model = RewardModel.from_pretrained(
        FLAGS.gold_reward_model_path,
        flash_attn=FLAGS.flash_attn,
        mixed_precision=True,
        torch_dtype=torch.float16,
        cache_dir=None,
        low_cpu_mem_usage=True,
        config=RewardConfig(backbone_model_name_or_path=FLAGS.sft10k_path),
    )
    gold_model.eval()
    print("Loaded gold reward model")
    
    @torch.no_grad()
    def get_gold_score(completions, shard_size=1):
        if shard_size >= 1:
            scores, beg = [], 0
            with tqdm(total=len(completions), desc="Computing gold scores") as pbar:
                while beg < len(completions):
                    end = min(beg+shard_size, len(completions))
                    tokenized_completions = gold_tokenizer(
                        completions[beg:end],
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    tokenized_completions = accelerate.utils.send_to_device(tokenized_completions, trainer.accelerator.device)
                    torch.cuda.empty_cache()
                    scores.append(gold_model(**tokenized_completions, return_dict=False)[0])
                    pbar.update(end-beg)
                    beg = end
            concated_scores = torch.cat(scores, dim=0)
            assert concated_scores.shape[0] == len(completions) and concated_scores.ndim == 1
            return concated_scores
        else:
            tokenized_completions = gold_tokenizer(
                completions,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_completions = accelerate.utils.send_to_device(tokenized_completions, trainer.accelerator.device)
            scores = gold_model(**tokenized_completions, return_dict=False)
            assert concated_scores.shape[0] == len(completions) and concated_scores.ndim == 1
            return scores

    
    def save_model(checkpoint_dir, epoch_num, add_prefix=True):
        if add_prefix:
            checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        if trainer.accelerator.is_main_process:
            trainer.accelerator.unwrap_model(model).save_pretrained(
                checkpoint_dir,
                save_function=trainer.accelerator.save,
                is_main_process=trainer.accelerator.is_main_process,
                state_dict=trainer.accelerator.get_state_dict(model),
            )
            if trainer.accelerator.is_main_process:
                tokenizer.save_pretrained(checkpoint_dir)
            trainer.accelerator.print(f"Checkpointing Epoch {epoch_num} -> {checkpoint_dir}")
 
    pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=max(batch_size_pref_data, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        eval_pref_dataset,
        batch_size=max(batch_size_pref_data, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )

    last_epoch = -1

    if batch_size_online_data == 0:
        zipped_dataloaders = pref_dataset_dataloader
    elif batch_size_pref_data == 0:
        zipped_dataloaders = trainer.dataloader
    else:
        zipped_dataloaders = zip(trainer.dataloader, pref_dataset_dataloader)
        
    if FLAGS.use_gold_reward_model:
        rew_fn = lambda x: get_gold_score(completions=x, shard_size=FLAGS.gold_shard_size)
    else:
        rew_fn = get_pred_reward

    for epoch, batches in tqdm(enumerate(zipped_dataloaders)):
        if isinstance(batches, tuple) and len(batches) == 2:
            batch, pref_batch = batches
        else:
            if batch_size_online_data == 0:
                batch, pref_batch = None, batches
            elif batch_size_pref_data == 0:
                batch, pref_batch = batches, None
        
        if batch is not None:
            #### Construct query tensors
            query_tensors = tokenizer(batch["query"], padding=True, truncation=True, max_length=128, return_tensors='pt')
            query_tensors = accelerate.utils.send_to_device(query_tensors, trainer.accelerator.device)

            #### Get generations from SFTModel (including prompt)
            if query_tensors.input_ids.shape[0] > FLAGS.max_gen_batch_size:
                generation_tokens = []
                for i in tqdm(range(0, query_tensors.input_ids.shape[0], FLAGS.max_gen_batch_size), desc=f"Generating for epoch {epoch}"):
                    generation_tokens.append(trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors[i:i+FLAGS.max_gen_batch_size], **generation_kwargs))
                    torch.cuda.empty_cache()
                generation_tokens = torch.cat(generation_tokens, dim=0)
            else:
                generation_tokens = trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors, **generation_kwargs)
            texts = tokenizer.batch_decode(generation_tokens, skip_special_tokens=True)

            #### Update batch with response
            batch["response"] = [x.split(ASSISTANT_TOKEN)[-1] for x in texts]
            response_tensors = tokenizer(batch["response"], padding=True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            response_tensors = [response_tensors[i] for i in range(0, len(response_tensors))]

            #### Compute reward score
            if FLAGS.use_gold_reward_model:
                gold_model = trainer.accelerator.prepare_model(gold_model)
            rewards = rew_fn(texts)
            rewards = [rewards[i] for i in range(0, len(rewards))]
            torch.cuda.empty_cache()
            
            ### Reprocess query tensors
            query_tensors = query_tensors.input_ids
            query_tensors = [query_tensors[i] for i in range(0, len(query_tensors))]
        else:
            query_tensors, response_tensors, rewards = [], [], []
        
        if pref_batch is not None:
            ### Process preference dataset
            pref_query = tokenizer(pref_batch["query"], padding=True, truncation=True, max_length=128, return_tensors='pt').input_ids
            pref_query_tensors = accelerate.utils.send_to_device(pref_query, trainer.accelerator.device)
            
            pref_response_tensors = tokenizer(pref_batch["response"], padding=True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            pref_response_tensors = accelerate.utils.send_to_device(pref_response_tensors, trainer.accelerator.device)
            
            pref_texts = pref_batch["text"]
            pref_rewards = rew_fn(pref_texts)
            
            # now append the preference dataset to the batch
            query_tensors.extend([pref_query_tensors[i] for i in range(0, len(pref_query_tensors))])
            response_tensors.extend([pref_response_tensors[i] for i in range(0, len(pref_response_tensors))])
            rewards.extend([pref_rewards[i] for i in range(0, len(pref_rewards))])
            
        if FLAGS.use_gold_reward_model:
            gold_model.to(torch.device("cpu"))

        #### Run RBC step
        stats = trainer.step(query_tensors, response_tensors, rewards)
        trainer.log_stats(stats, batch or {}, rewards)
        
        if epoch > 0 and epoch > last_epoch:
            save_model(model_name + f"_epoch_{epoch}", epoch)
            last_epoch = epoch

    #### Save model
    save_model(model_name, -1)

if __name__ == "__main__":
    app.run(main)