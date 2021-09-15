import torch
import os
import numpy as np
from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
import copy
import math
import pickle
def load_checkpoint_to_cpu(path):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
    state = torch.load(
        path, map_location='cpu',
        #, map_location=lambda s, l: default_restore_location(s, 'cpu'),
    )
    #state = _upgrade_state_dict(state)
    return state

def get_encoder_module():
    module = [    "encoder.embed_tokens.weight", 
                "encoder.layers.0.self_attn.in_proj_weight",
                "encoder.layers.0.self_attn.in_proj_bias",
                "encoder.layers.0.self_attn.out_proj.weight",
                "encoder.layers.0.self_attn.out_proj.bias",
                "encoder.layers.0.self_attn_layer_norm.weight",
                "encoder.layers.0.self_attn_layer_norm.bias",
                "encoder.layers.0.fc1.weight",
                "encoder.layers.0.fc1.bias",
                "encoder.layers.0.fc2.weight",
                "encoder.layers.0.fc2.bias",
                "encoder.layers.0.final_layer_norm.weight",
                "encoder.layers.0.final_layer_norm.bias",
                "encoder.layers.1.self_attn.in_proj_weight",
                "encoder.layers.1.self_attn.in_proj_bias",
                "encoder.layers.1.self_attn.out_proj.weight",
                "encoder.layers.1.self_attn.out_proj.bias",
                "encoder.layers.1.self_attn_layer_norm.weight",
                "encoder.layers.1.self_attn_layer_norm.bias",
                "encoder.layers.1.fc1.weight",
                "encoder.layers.1.fc1.bias",
                "encoder.layers.1.fc2.weight",
                "encoder.layers.1.fc2.bias",
                "encoder.layers.1.final_layer_norm.weight",
                "encoder.layers.1.final_layer_norm.bias",
                "encoder.layers.2.self_attn.in_proj_weight",
                "encoder.layers.2.self_attn.in_proj_bias",
                "encoder.layers.2.self_attn.out_proj.weight",
                "encoder.layers.2.self_attn.out_proj.bias",
                "encoder.layers.2.self_attn_layer_norm.weight",
                "encoder.layers.2.self_attn_layer_norm.bias",
                "encoder.layers.2.fc1.weight",
                "encoder.layers.2.fc1.bias",
                "encoder.layers.2.fc2.weight",
                "encoder.layers.2.fc2.bias",
                "encoder.layers.2.final_layer_norm.weight",
                "encoder.layers.2.final_layer_norm.bias",
                ]
    return module

def get_decoder_module():
    module = [    "decoder.embed_out",
                "decoder.embed_tokens.weight",
                "decoder.layers.0.self_attn.in_proj_weight",
                "decoder.layers.0.self_attn.in_proj_bias",
                "decoder.layers.0.self_attn.out_proj.weight",
                "decoder.layers.0.self_attn.out_proj.bias",
                "decoder.layers.0.self_attn_layer_norm.weight",
                "decoder.layers.0.self_attn_layer_norm.bias",
                "decoder.layers.0.encoder_attn.in_proj_weight", #
                "decoder.layers.0.encoder_attn.in_proj_bias",
                "decoder.layers.0.encoder_attn.out_proj.weight", #
                "decoder.layers.0.encoder_attn.out_proj.bias",
                "decoder.layers.0.encoder_attn_layer_norm.weight",
                "decoder.layers.0.encoder_attn_layer_norm.bias",
                "decoder.layers.0.fc1.weight", #
                "decoder.layers.0.fc1.bias",   
                "decoder.layers.0.fc2.weight", #
                "decoder.layers.0.fc2.bias",
                "decoder.layers.0.final_layer_norm.weight",
                "decoder.layers.0.final_layer_norm.bias",
                "decoder.layers.1.self_attn.in_proj_weight",
                "decoder.layers.1.self_attn.in_proj_bias",
                "decoder.layers.1.self_attn.out_proj.weight",
                "decoder.layers.1.self_attn.out_proj.bias",
                "decoder.layers.1.self_attn_layer_norm.weight",
                "decoder.layers.1.self_attn_layer_norm.bias",
                "decoder.layers.1.encoder_attn.in_proj_weight",
                "decoder.layers.1.encoder_attn.in_proj_bias",
                "decoder.layers.1.encoder_attn.out_proj.weight",
                "decoder.layers.1.encoder_attn.out_proj.bias",
                "decoder.layers.1.encoder_attn_layer_norm.weight",
                "decoder.layers.1.encoder_attn_layer_norm.bias",
                "decoder.layers.1.fc1.weight",
                "decoder.layers.1.fc1.bias",
                "decoder.layers.1.fc2.weight",
                "decoder.layers.1.fc2.bias",
                "decoder.layers.1.final_layer_norm.weight",
                "decoder.layers.1.final_layer_norm.bias",
                "decoder.layers.2.self_attn.in_proj_weight",
                "decoder.layers.2.self_attn.in_proj_bias",
                "decoder.layers.2.self_attn.out_proj.weight",
                "decoder.layers.2.self_attn.out_proj.bias",
                "decoder.layers.2.self_attn_layer_norm.weight",
                "decoder.layers.2.self_attn_layer_norm.bias",
                "decoder.layers.2.encoder_attn.in_proj_weight",
                "decoder.layers.2.encoder_attn.in_proj_bias",
                "decoder.layers.2.encoder_attn.out_proj.weight",
                "decoder.layers.2.encoder_attn.out_proj.bias",
                "decoder.layers.2.encoder_attn_layer_norm.weight",
                "decoder.layers.2.encoder_attn_layer_norm.bias",
                "decoder.layers.2.fc1.weight",
                "decoder.layers.2.fc1.bias",
                "decoder.layers.2.fc2.weight",
                "decoder.layers.2.fc2.bias",
                "decoder.layers.2.final_layer_norm.weight",
                "decoder.layers.2.final_layer_norm.bias",

                ]
    return module


def get_module():

    res = ["encoder.embed_tokens","decoder.embed_tokens"]

    p1 = ["encoder.layers.", "decoder.layers."]
    p2 = ["0.", "1.", "2.", "3.", "4.", "5."]
    p3 = ["self_attn.in_q", "self_attn.in_k", "self_attn.in_v", "self_attn.out_proj", "self_attn_layer_norm",
             "fc1", "fc2", "final_layer_norm",
            "encoder_attn.in_q", "encoder_attn.in_k", "encoder_attn.in_v", "encoder_attn.out_proj", "encoder_attn_layer_norm"]
    p4 = ["self_attn.in_q", "self_attn.in_k", "self_attn.in_v", "self_attn.out_proj", "self_attn_layer_norm",
             "fc1", "fc2", "final_layer_norm"]
    for a in p1:
        for b in p2:
            if a == "encoder.layers.":
                for c in p4:
                    res.append(a + b + c)
            else:
                for c in p3:
                    res.append(a + b + c)
    return res




def hook_fn_forward(module, input, output):
    #print(module.__name__) # 用于区分模块
    # 首先打印出来
    #print('output', output.size())
    if module.__name__ == 'decoder':
        tmp_feat_out['decoder.embed_out'] = output[0].data if isinstance(output, tuple) else output.data
    else:
        tmp_feat_out[module.__name__] = output[0].data if isinstance(output, tuple) else output.data

def hook_fn_backward(module, grad_input, grad_output):
    #print(module.__name__) # 为了区分模块
    # 打印 grad_output
    #tt = [grad_output[i].size() for i in range(len(grad_output))]
    #print('grad_output', tt)
    #print('grad_output', grad_output[0].size()) 
    if module.__name__ == 'decoder':
        tmp_grad_out['decoder.embed_out'] = grad_output[0].data if isinstance(grad_output, tuple) else grad_output.data
    else:
        tmp_grad_out[module.__name__] = grad_output[0].data if isinstance(grad_output, tuple) else grad_output.data

tmp_grad_out = {}
tmp_feat_out = {}



def validate(args, trainer, task, epoch_itr, subsets):
    subset = subsets[0]
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
        prefix='valid on \'{}\' subset'.format(subset),
        no_progress_bar='simple'
    )
    # compute the fisher information matrix
    trainer.model.eval()
    trainer.criterion.eval()
    module_list = get_module()
    module_list.append('decoder')
    modules = trainer.model.named_modules()
    for name, module in modules:
        name = name.split(".", 2)
        if len(name) > 2:
            name = name[2]
        else:
            continue
        if name in module_list:
            module.__name__ = name
            module.register_forward_hook(hook_fn_forward)
            module.register_backward_hook(hook_fn_backward)
            #module.__name__ = name

    
    taylor = {}
    module_list1 = get_module()
    module_list1.append('decoder.embed_out')
    for sample in progress:
        samples = trainer._prepare_sample(sample)
        for i in samples:
            if i != lang_str:
                continue
            sample = samples[i]
            loss, sample_size, logging_output = trainer.criterion(trainer.model, sample) 
            trainer.optimizer.backward(loss)
            #fisher_matrix = {}
            modules = trainer.model.named_modules()
            for name in module_list1:
                if name == 'encoder.embed_tokens' or name == 'decoder.embed_tokens':
                    continue
                if name in taylor.keys():
                    #print(name, tmp_grad_out[name].size(),  tmp_feat_out[name].size())

                    # Choose calculation method: Taylor Expansion or absolute value
                    tmp = (tmp_grad_out[name] * tmp_feat_out[name]).abs() # The Taylor Expansion
                    #tmp = tmp_feat_out[name].abs().float() # The absolute value
                    tmp = tmp.view(-1, tmp.size(-1)).mean(0)
                    taylor[name] += tmp
                else:
                    #print(name, tmp_grad_out[name].size(),  tmp_feat_out[name].size())
                    tmp = (tmp_grad_out[name] * tmp_feat_out[name]).abs()
                    #tmp = tmp_feat_out[name].abs().float()
                    tmp = tmp.view(-1, tmp.size(-1)).mean(0)
                    taylor[name] = tmp

        trainer.zero_grad()
    return taylor



def main(args):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=True, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    filename = args.path_baseline
    if not os.path.exists(filename):
        raise IOError('Model file not found: {}'.format(filename))
    state = checkpoint_utils.load_checkpoint_to_cpu(filename)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')


    # compute importance value
    taylor = validate(args, trainer, task, epoch_itr, valid_subsets)

    value_save = open('importance_value/%s'%(lang_str), 'wb')
    pickle.dump(taylor, value_save)


if __name__ == "__main__":
    parser = options.get_training_parser()
    parser.add_argument('--focus-lang')
    args = options.parse_args_and_arch(parser)
    lang_str = args.focus_lang
    main(args)
