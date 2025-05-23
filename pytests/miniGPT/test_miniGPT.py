from mpi4py import MPI
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel
import deepspeed
from torch_setup import get_device, get_device_type, init_distributed,  get_profiler_activities
import json
import argparse

# 2.1 Synthetic Dataset
class SyntheticTextDataset(Dataset):
    def __init__(self, seq_len, vocab_size, num_samples):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random token sequence, shifted for labels
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}

# 2.2 Model Factory
def build_model(vocab_size, seq_len, n_layers=32, hidden_size=512, n_heads=8):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,
        n_ctx=seq_len,
        n_embd=hidden_size,
        n_layer=n_layers,
        n_head=n_heads
    )
    return GPT2LMHeadModel(config)


def train(model, loader, epochs, verbose=False):
    for epoch in range(3):
        for step, batch in enumerate(loader):
            inputs = batch["input_ids"].to(model.local_rank)
            labels = batch["labels"].to(model.local_rank)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            model.backward(loss)
            model.step()
            if verbose and step % 20 == 0:
                print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train miniGPT with configurable parallelism")
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Tensor parallelism degree (overrides config.json if set)")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None,
                        help="Pipeline parallelism degree (overrides config.json if set)")
    parser.add_argument("--num-layers", type=int, default=32,
                        help="Number of Transformer layers")
    parser.add_argument("--micro-batch-size", type=int, default=8,
                        help="Micro batch size per GPU")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Epochs")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--trace-dir", type=str, default="miniGPT_trace")

    args = parser.parse_args()

    # 1. Initialize DeepSpeed
    ds_config = "ds_config.json"
    #world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dist, rank, world_size = init_distributed()
    #rank = int(os.environ.get("RANK", "0"))

    # 2. Prepare dataset & dataloader
    seq_len = args.sequence_length
    vocab_size = 50257
    dataset = SyntheticTextDataset(seq_len, vocab_size, num_samples=10000)
    loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=True)

    # 3. Build model & wrap with DeepSpeed
    model = build_model(vocab_size, seq_len, n_layers=args.num_layers)
    # Load DeepSpeed config and ensure tensor_parallel is a dict
    with open(ds_config, 'r') as fp:
        ds_config_dict = json.load(fp)
    # Override DeepSpeed config with CLI args if provided
    if args.tensor_parallel_size is not None:
        # apply to 'tensor_parallel' shorthand
        ds_config_dict['tp'] = {'tp_size': args.tensor_parallel_size, 'enabled': True}
    if args.pipeline_parallel_size is not None:
        # ensure pipeline dict exists
        pipeline_cfg = ds_config_dict.get('pipeline', {})
        pipeline_cfg['stages'] = args.pipeline_parallel_size
        pipeline_cfg['enabled'] = True
        ds_config_dict['pipeline'] = pipeline_cfg
    # Handle tensor parallel shorthand or dict under 'tensor_parallel'
    tp_cfg = ds_config_dict.pop('tensor_parallel', None)
    if tp_cfg is not None:
        if isinstance(tp_cfg, int):
            tp_cfg = {'tp_size': tp_cfg, 'enabled': True}
        # Use alias 'tp' for DeepSpeedTPConfig
        ds_config_dict['tp'] = tp_cfg
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=ds_config_dict,
        model_parameters=model.parameters()
    )

    # 4. Training loop
    model.train()
    if args.profile:
        with profile(
            activities=get_profiler_activities(),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            train(model, loader, epochs=args.epochs, verbose=(rank==0))
            os.makedirs(args.trace_dir, exist_ok=True)
            prof.export_chrome_trace(f"{args.trace_dir}/trace-{rank}-of-{world_size}.json")
    else:
        train(model, loader, epochs=args.epochs, verbose=(rank==0))

if __name__ == "__main__":
    main()
