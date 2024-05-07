import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import Split
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.global_vars import set_global_variables
from fmoe.megatron.layers import fmoefy

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)


    
def model_provider(args):
    """Build the model."""

    transformer_config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32)

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=100,
        max_sequence_length=args.seq_length)

    
    gpt_model = fmoefy(gpt_model, fmoe_num_experts=args.moe_num_expert)

    return gpt_model

def get_train_data_iterator(args):
    config = GPTDatasetConfig(
        is_built_on_rank=lambda:(parallel_state.is_pipeline_last_stage() or parallel_state.is_pipeline_first_stage()),
        random_seed=0,
        sequence_length=args.tgt_len,
        blend=[],
        mock=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer="dummy")

    training_data = MockGPTDataset(Split.train, config)

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)

    train_iterator = iter(train_dataloader)
    return train_iterator

def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = data['tokens'].to(device)
    attention_mask = data['attention_mask'].to(device)
    position_ids = data['position_ids'].to(device)
    labels = data['labels'].to(device)
    loss_mask = data['loss_mask'].to(device)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)

def save_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict = gpt_model.sharded_state_dict(prefix='')
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(checkpoint_path, gpt_model):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model

if __name__ == "__main__":
    # Parse command-line arguments
    from megatron.arguments import parse_args
    from argparse import Namespace

    # Create your Namespace object
    args = Namespace(
        micro_batch_size=4,
        num_layers=3,
        hidden_size=8,
        num_attention_heads=2,
        max_position_embeddings=512,
        tokenizer_type='BertWordPieceTokenizer',
        fp16=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        lr=0.25,
        seq_length=512,
        moe_num_expert = 8
    )

    # Convert Namespace to dictionary
    defaults = vars(args)

    # Pass the dictionary to parse_args
    # parsed_args = parse_args(extra_args_provider=None, defaults=defaults)

    set_global_variables(extra_args_provider=None, args_defaults=defaults)

    # initialize_distributed()
    initialize_distributed(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    gpt_model = model_provider(args)
    device = torch.device("cuda")
    gpt_model.to(device)

    optim = Adam(gpt_model.parameters(), lr=args.lr)

    train_iterator = get_train_data_iterator()

    forward_backward_func = get_forward_backward_func()

    # Running the model for the desired number of iterations
    for _ in range(args.max_steps):
        optim.zero_grad()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=args.batch_size // args.micro_batch_size,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.seq_length,
            forward_only=False)

        optim.step()

        print(f'Losses reduced :  {losses_reduced}')

    # Saving the model
    ckpt_path = os.getcwd() + '/ckpt'
    Path(ckpt_path).mkdir(exist_ok=True)
    save_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)

    # Loading the model
    gpt_model = load_distributed_checkpoint(gpt_model=gpt_model, checkpoint_path=ckpt_path)
    gpt_model.to(device)
    print('Successfully loaded the model')