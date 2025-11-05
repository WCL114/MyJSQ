import torch
from jsq.smooth import smooth_layer
from jsq.quantize import quantize_layer
from jsq.data import get_loaders
from jsq.layerwrapper import WrappedGPT
from jsq.utils import find_layers, prepare_calibration_input, clip_matrix, generate_ss, edit_activation_minmax
import math
import copy as _copy
from typing import Dict, List

# --- BEGIN IMPLEMENTATION ---
@torch.no_grad()
def _evaluate_lm_loss(model, eval_input_ids: torch.Tensor, device, max_tokens: int = 4096) -> float:
    """
    使用交叉熵loss评估语言模型在给定输入上的拟合效果，用于模拟退火中的P(COM(LLM;V))。
    只做前向，labels=inputs（标准LM困惑度评估做法），返回平均loss（越小越好）。
    """
    model.eval()
    input_ids = eval_input_ids.to(device)
    # 截断评估token数，避免搜索阶段过慢
    if input_ids.numel() > max_tokens:
        input_ids = input_ids[:, :max_tokens]

    # 将长序列拆成若干 seqlen 窗口求平均loss，避免显存峰值
    seqlen = getattr(model.config, "max_position_embeddings", 2048)
    seqlen = max(256, min(seqlen, 2048))  # 保守上限
    total_loss = 0.0
    total_tokens = 0

    for start in range(0, input_ids.shape[1], seqlen):
        chunk = input_ids[:, start:start+seqlen]
        if chunk.shape[1] < 2:
            break
        outputs = model(input_ids=chunk, labels=chunk)
        loss = outputs.loss.detach().float().item()
        # 有效token按 (len-1) 计，接近 huggingface 计算方式
        total_loss += loss * (chunk.shape[1] - 1)
        total_tokens += (chunk.shape[1] - 1)

    if total_tokens == 0:
        return float("inf")
    return total_loss / total_tokens


def _parse_r_choices(args) -> List[float]:
    """
    解析命令行传入的候选r集合字符串。
    """
    raw = getattr(args, "sa_r_choices", None)
    if not raw:
        return [0.0, 4e-5, 5e-5, 6e-5, 7e-5]
    items = []
    for s in raw.split(","):
        s = s.strip()
        if s.lower().endswith("e-5") or "e" in s.lower():
            items.append(float(s))
        else:
            items.append(float(s))
    # 去重并排序
    vals = sorted(set(items))
    return vals if len(vals) > 0 else [0.0, 4e-5, 5e-5, 6e-5, 7e-5]


def _collect_layer_names(model) -> List[str]:
    """
    收集本工程实际压缩遍历到的block层名列表，用于构造编辑向量V。
    逻辑与 joint_pq 中的层选择保持一致（LLaMA/Falcon/ChatGLM等）。
    """
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True

    if CHATGLM:
        layers = model.transformer.encoder.layers
        prefix = 'transformer.encoder.layers'
    elif Falcon:
        layers = model.transformer.h
        prefix = 'transformer.h'
    else:
        layers = model.model.layers
        prefix = 'model.layers'

    names = [f"{prefix}.{i}" for i in range(len(layers))]
    return names


def _compress_once_with_r_map(args, model, tokenizer, r_map: Dict[str, float], device) -> None:
    """
    用给定的 per-layer r_map 执行一次完整的 统计→剪枝→平滑→量化 流程。
    这是 joint_pq 主循环的“无搜索”变体：唯一差别是 add_batch 时使用 r_map[layer_name] 来做编辑，而不是 args.r。
    注意：此函数对 model 就地修改（in-place）。
    """
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
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
        if CHATGLM:
            layer_name = f'transformer.encoder.layers.{i}'
        elif Falcon:
            layer_name = f'transformer.h.{i}'
        else:
            layer_name = f'model.layers.{i}'

        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and layer_name in model.hf_device_map:
            dev = model.hf_device_map[layer_name]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            )

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

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

        # 基于 r_map 的逐层激活编辑
        per_layer_r = float(r_map.get(layer_name, 0.0))

        def add_batch(name):
            def tmp(_, inp, out):
                x = inp[0].data
                if per_layer_r is not None and per_layer_r > 0:
                    x = edit_activation_minmax(x, r=per_layer_r, dim=-1)
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

        # 剪枝
        for name in subset:
            weight = torch.abs(subset[name].weight.data)
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)
            W_metric = weight * activation
            W_metric = W_metric + args.rho * ss

            W_mask = (torch.zeros_like(W_metric) == 1)
            if getattr(args, "sparsity_type", "unstructured") not in ["4:8", "2:4"] and getattr(args, "sparsity_ratio", 0) > 0:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
            else:
                # 若指定结构化稀疏，沿用原逻辑（此处按未传 prune_n/m 的简单情形）
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        # 重新前向一次，为平滑统计输出
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

        # 平滑 & 量化
        smooth_layer(layer_name, layer, act_scales, 0.5)
        # --- Start of feature modification ---
        # Support separate weight and activation bit widths
        weight_nbits = getattr(args, 'weight_nbits', None)
        act_nbits = getattr(args, 'act_nbits', None)
        nbits = getattr(args, 'nbits', None)
        quantize_layer(layer, nbits=nbits, weight_nbits=weight_nbits, act_nbits=act_nbits)
        # --- End of feature modification ---

        # 交替输入输出（与原 joint_pq 一致）
        inps, outs = outs, inps

    torch.cuda.empty_cache()


def _simulated_annealing_search(args, model, tokenizer, device) -> Dict[str, float]:
    """
    基于模拟退火搜索 per-layer 的激活编辑强度 r_map。
    返回: { 'model.layers.i': r_value }
    """
    # 1) 收集可编辑层名与候选集合
    layer_names = _collect_layer_names(model)
    R = _parse_r_choices(args)
    if len(R) == 0:
        R = [0.0]

    # 2) 准备评估数据（使用 get_loaders 返回的 validation/c4）
    _, valenc = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)
    # valenc 可能是 TokenizerWrapper
    if hasattr(valenc, "input_ids"):
        eval_input_ids = valenc.input_ids
    else:
        eval_input_ids = valenc

    # 3) 初始化编辑向量（随机）
    import random
    random.seed(args.seed)
    curr_r_map = {name: random.choice(R) for name in layer_names}

    # 保存最优
    best_r_map = dict(curr_r_map)
    # 深拷贝模型，基于当前解压缩并评估
    base_device = device
    curr_model = _copy.deepcopy(model).to(base_device)
    _compress_once_with_r_map(args, curr_model, tokenizer, curr_r_map, base_device)
    curr_loss = _evaluate_lm_loss(curr_model, tokenizer, eval_input_ids, base_device, max_tokens=getattr(args, "sa_eval_tokens", 4096))
    curr_P = -curr_loss
    best_P = curr_P
    del curr_model
    torch.cuda.empty_cache()

    # 4) 模拟退火参数
    Ti = float(getattr(args, "sa_Ti", 300.0))
    Tf = float(getattr(args, "sa_Tf", 10.0))
    steps = int(getattr(args, "sa_steps", 50))
    steps = max(1, steps)
    # 指数退火：T_k = Ti * (Tf/Ti)^(k/steps)
    def temperature(step):
        ratio = (Tf / Ti) ** (step / steps)
        return Ti * ratio

    # 5) 退火主循环
    for step in range(steps):
        T = temperature(step)
        # 随机挑一个层，随机换一个不同的候选r
        l_name = random.choice(layer_names)
        old_r = curr_r_map[l_name]
        cand_choices = [v for v in R if v != old_r]
        if len(cand_choices) == 0:
            continue
        new_r = random.choice(cand_choices)

        prop_r_map = dict(curr_r_map)
        prop_r_map[l_name] = new_r

        # 用候选向量复制并压缩评估
        prop_model = _copy.deepcopy(model).to(base_device)
        _compress_once_with_r_map(args, prop_model, tokenizer, prop_r_map, base_device)
        prop_loss = _evaluate_lm_loss(prop_model, tokenizer, eval_input_ids, base_device, max_tokens=getattr(args, "sa_eval_tokens", 4096))
        prop_P = -prop_loss
        del prop_model
        torch.cuda.empty_cache()

        # Metropolis 准则
        dP = prop_P - curr_P
        accept = dP >= 0 or math.exp(dP / max(T, 1e-6)) > random.random()
        if accept:
            curr_r_map = prop_r_map
            curr_P = prop_P

        # 记录全局最优
        if prop_P > best_P:
            best_P = prop_P
            best_r_map = prop_r_map

    return best_r_map
# --- END IMPLEMENTATION ---


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

    r_map = None
    if getattr(args, "sa_search", False):
        print("[JSQ] Start simulated-annealing search for activation editing vector (per-layer r)...")
        r_map = _simulated_annealing_search(args, model, tokenizer, device)
        unique_vals = {}
        for _, v in r_map.items():
            unique_vals[v] = unique_vals.get(v, 0) + 1
        print(f"[JSQ] Search finished. r distribution: {unique_vals}")

        # 用最优 r_map 对“主模型”做一次真正压缩并返回
        _compress_once_with_r_map(args, model, tokenizer, r_map, device)
        torch.cuda.empty_cache()
        return model
    else:
        if CHATGLM:
            layers = model.transformer.encoder.layers
        elif Falcon:
            layers = model.transformer.h
        else:
            layers = model.model.layers

        for i in range(len(layers)):
            layer = layers[i]
            if CHATGLM:
                layer_name = f'transformer.encoder.layers.{i}'
            elif Falcon:
                layer_name = f'transformer.h.{i}'
            else:
                layer_name = f'model.layers.{i}'

            subset = find_layers(layer)

            if hasattr(model, "hf_device_map") and layer_name in model.hf_device_map:
                dev = model.hf_device_map[layer_name]
                inps, outs, attention_mask, position_ids = (
                    inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
                )

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
                    # 优先使用搜索得到的 per-layer r；若未搜索则退回到原有 args.r/clip 行为
                    if r_map is not None:
                        per_layer_r = float(r_map.get(layer_name, 0.0))
                        if per_layer_r > 0:
                            x = edit_activation_minmax(x, r=per_layer_r, dim=-1)
                        else:
                            x = clip_matrix(x, args.abs, 0, args.clip_h)
                    else:
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

            # --- Start of feature modification ---
            print(f"quantizing layer {i}")
            # Support separate weight and activation bit widths
            weight_nbits = getattr(args, 'weight_nbits', None)
            act_nbits = getattr(args, 'act_nbits', None)
            nbits = getattr(args, 'nbits', None)
            quantize_layer(layer, nbits=nbits, weight_nbits=weight_nbits, act_nbits=act_nbits)
            # --- End of feature modification ---

            inps, outs = outs, inps

    torch.cuda.empty_cache()

    return model