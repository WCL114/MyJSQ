import torch
from jsq.smooth import smooth_layer
from jsq.quantize import quantize_layer
from jsq.data import get_loaders
from jsq.layerwrapper import WrappedGPT
from jsq.utils import find_layers, prepare_calibration_input, clip_matrix, generate_ss, edit_activation_minmax
import torch

def joint_pq(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True

    print(model)
    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        # 对 Llama 模型生成 rotary position embeddings
        if hasattr(model.model, "rotary_emb"):
            position_embeddings = model.model.rotary_emb(inps, position_ids)
        else:
            position_embeddings = None


    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        layer_name = f'model.layers.{i}'

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        print(f"get scales of layer {i} and pruning")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()

            key = layer_name + '.' + name
            if key in act_scales:
                act_scales[key] = torch.max(act_scales[key], comming_max)
            else:
                act_scales[key] = comming_max


        def add_batch(name):
            def tmp(_, inp, out):
                x = inp[0].data
                if getattr(args, "r", None) is not None and args.r > 0:
                    x = edit_activation_minmax(x, r=args.r, dim=-1)
                else:
                    x = clip_matrix(x, args.abs, 0, args.clip_h)
                stat_tensor(name, x)
                wrapped_layers[name].add_batch(x, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings
                    )[0]

        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)
            W_metric = weight * activation
            W_metric = W_metric + args.rho * ss

            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                elif Falcon:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=None)[0]
                else:
                    outs[j] = layer(
                        inps[j].unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings
                    )[0]


 
        print(f"smoothing layer {i}")
        smooth_layer(layer_name, layer, act_scales, 0.5) 

        print(f"quantizing layer {i}")
        quantize_layer(layer, nbits=args.nbits)

        inps, outs = outs, inps

    torch.cuda.empty_cache()

    return model