import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling

from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator

from transformers import get_scheduler

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict(
    {
        "train": ds_train.shuffle(seed=12).select(range(80000)),
        "valid": ds_valid.shuffle(seed=12).select(range(500))
    }
)

#DatasetDict({
#    train: Dataset({
#        features: ['input_ids'],
#        num_rows: 2203760
#    })
#    valid: Dataset({
#        features: ['input_ids'],
#        num_rows: 12763
#    })
#})

def get_dataloaders(accelerator: Accelerator, batch_size: int = 32):
    
    context_length = 128
    tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

    def tokenize(element):
        outputs = tokenizer(
            element['content'],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch =  [
            input_ids for length, input_ids in zip(outputs['length'], outputs['input_ids']) if length == context_length
        ]
        return {"input_ids": input_batch}

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)

    tokenized_datasets.set_format("torch")
    train_dataloader = DataLoader(tokenized_datasets["train"], 
                              batch_size=batch_size,
                              shuffle=True,
                        )
    eval_dataloader = DataLoader(tokenized_datasets["valid"], 
                             batch_size=batch_size,
                        )

    return train_dataloader, eval_dataloader, tokenizer

def training_function(config, args):

    batch_size = 32
    
    accelerator = Accelerator(mixed_precision="fp16")    
    #accelerator = Accelerator() 
    train_dataloader, eval_dataloader, tokenizer = get_dataloaders(accelerator, batch_size=batch_size)

    keywords = ["plt","pd","sk","fit","predict"," plt"," pd"," sk"," fit"," predict"]
    keytoken_ids = [tokenizer(k).input_ids[0] for k in keywords]

    def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
        shift_labels = inputs[..., 1:].contiguous() # first token removed
        #shift_labels = logits[..., :-1, :].contiguous()
        shift_logits = logits[..., :-1, :].contiguous() # last logit removed
        # 토큰당 손실값 계산
        #loss_fct = CrossEntropyLoss(reduce=False)
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # 샘플당 손실값을 resize하고 평균화
        loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
        # Calculate and scale weighting
        weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2])
        weights = alpha * (1.0 + weights)
        # Calculate weighted average
        weighted_loss = (loss_per_sample * weights).mean()
        return weighted_loss

    weight_decay = 0.1

    def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
        params_with_wd, params_without_wd = [], []
        for n, p in model.named_parameters():
            if any(nd in n for nd in no_decay):
                params_without_wd.append(p)
            else:
                params_with_wd.append(p)
        return [
            {"params": params_with_wd, "weight_decay": weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    def evaluate():
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            #print(batch["input_ids"].shape) #torch.Size([32, 128])
            with torch.no_grad():
                outputs = model(batch["input_ids"], labels=batch["input_ids"])

            loss = outputs.loss
            batch_size = len(batch["input_ids"])
            losses.append(accelerator.gather(loss.repeat(batch_size)))
            #losses.append(accelerator.gather(outputs.loss.item()))
            #print(outputs.loss.shape)
            #print(len(losses)) #1,2,3,....100 ==> 12763 / (32 * 4) = 99.7
        #print(losses)
        loss = torch.mean(torch.cat(losses))
        #loss = torch.mean(torch.tensor(losses))
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")
        return loss.item(), perplexity.item()

    context_length = 128
    
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer),
        n_ctx=context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    model = GPT2LMHeadModel(config)

    optimizer = AdamW(get_grouped_params(model), lr=5e-4)

    
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    gradient_accumulation_steps = 8
    
    num_train_epochs = 1
    num_update_steps_per_epoch = len(train_dataloader)
    #num_training_steps = num_train_epochs * num_update_steps_per_epoch
    num_training_steps = num_train_epochs * num_update_steps_per_epoch // gradient_accumulation_steps
    
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps,
    )

    #from huggingface_hub import Repository, get_full_repo_name

    #model_name = "codeparrot-ds-accelerate"
    #repo_name = get_full_repo_name(model_name)
    #repo_name

    output_dir = "codeparrot-ds-accelerate"
    #repo = Repository(output_dir, clone_from=repo_name)

    from tqdm.notebook import tqdm

    gradient_accumulation_steps = 8
    #eval_steps = 5000
    eval_steps = 100

    model.train()
    completed_steps = 0
    #samples_per_step = batch_size * gradient_accumulation_steps
    samples_per_step = batch_size

    num_training_steps = num_train_epochs * len(train_dataloader)  // gradient_accumulation_steps
    progress_bar = tqdm(range(num_training_steps))

    #print(f"length of train_dataloader: {len(train_dataloader)}") # 17217

    for epoch in range(num_train_epochs):
        #for step, batch in tqdm(enumerate(train_dataloader, start=1), total=num_training_steps):
        for step, batch in enumerate(train_dataloader, start=1):    
            logits = model(batch["input_ids"]).logits
            loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)
            if step % 800 == 0: #eval_step * gradient_accumulation_steps
                accelerator.print(
                    {
                        "lr": lr_scheduler.get_lr()[0],
                        "samples": step * samples_per_step,
                        "steps": completed_steps,
                        #"loss/train": loss.item() * gradient_accumulation_steps,
                        "loss/train": loss.item(),
                    }
                )
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)
            #samples_per_step += len(batch)
            if step % gradient_accumulation_steps == 0:    
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1
                progress_bar.update(1)
            if (step % (eval_steps * gradient_accumulation_steps)) == 0:
                eval_loss, perplexity = evaluate()
                accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
                model.train()
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                    #repo.push_to_hub(
                    #    commit_message=f"Training in progress step {step}", blocking=False
                    #)

#from huggingface_hub import Repository, get_full_repo_name

#model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
#repo_name = get_full_repo_name(model_name)
#repo_name
#repo = Repository(output_dir, clone_from=repo_name)
#notebook_launcher(function, args, num_processes, mixed_precision, use_port, master_addr, node_rank, num_nodes)

#from accelerate import notebook_launcher

#notebook_launcher(training_function, num_processes=4)

import argparse
 
def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {"lr": 2e-5, "num_epochs": 3, "seed": 42, "batch_size": 16}
    training_function(config, args)


if __name__ == "__main__":
    main()

