import torch
import argparse
from jsq.prune import joint_pq
from transformers import AutoTokenizer,AutoModelForCausalLM

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True)

    light_model = joint_pq(args, model, tokenizer)
    
    if args.save_model is not None:
        print(f"Saving pruned & quantized model to: {args.save_model}")
        light_model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print("Model and tokenizer saved successfully!")
    
    return light_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5, help='number of shots')
    parser.add_argument("--ngpu", "-g", type=int, default=8)
    parser.add_argument("--data_dir", "-d", type=str, default="data", required=True, help='dataset location')
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--path", type=str, required=False, help='model checkpoint location')

    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--cache_dir", default="/mnt/disk1/hg/huggingface/cache", type=str)

    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--clip_l', type=float, default=0.0)
    parser.add_argument('--clip_h', type=float, default=0.01)
    parser.add_argument('--abs', action="store_false")
    parser.add_argument('--rho', type=float, default=2.1)
    # --- Start of feature modification ---
    parser.add_argument("--nbits", type=int, default=None, help='Bit width for both weights and activations (backward compatibility). If specified, overrides weight_nbits and act_nbits.')
    parser.add_argument("--weight_nbits", type=int, default=8, help='Bit width for weight quantization (default: 8)')
    parser.add_argument("--act_nbits", type=int, default=8, help='Bit width for activation quantization (default: 8)')
    # --- End of feature modification ---
    # main.py
    parser.add_argument('--r', type=float, default=5e-5, help='activation edit strength (paper Eq.5); set to e.g. 5e-5')


    args = parser.parse_args()
    main(args)