from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
import fire

def compute_layer_requirements(config):
    """Calculate memory requirements for base model and per-layer components."""
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    vocab_size = config.vocab_size

    # Base components (embeddings, final norm, output layer)
    base_params = (
        vocab_size * hidden_size +  # token embeddings
        2 * hidden_size +           # final layer norm
        hidden_size * vocab_size    # output layer
    )

    # Per-layer components (attention, MLP, layer norms)
    layer_params = (
        4 * hidden_size**2 +       # attention projections (q,k,v,o)
        3 * hidden_size * intermediate_size +  # MLP layers
        4 * hidden_size             # layer norms (input+post attention)
    )

    return base_params, layer_params


def get_hf_model(model_id='meta-llama/Llama-2-7b-chat-hf', device='cuda', num_layers=None):
    """Load model with automatic layer adjustment based on available memory."""
    config = AutoConfig.from_pretrained(model_id,
                        trust_remote_code=True )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if num_layers is None and device == "cuda":
        try:
            # Calculate memory requirements
            base_params, layer_params = compute_layer_requirements(config)
            bytes_per_param = 2  # float16 precision
            base_mem = base_params * bytes_per_param
            per_layer_mem = layer_params * bytes_per_param

            # Get available memory
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            free_mem = total_mem - reserved_mem

            # Calculate maximum viable layers
            if free_mem < base_mem:
                raise RuntimeError("Insufficient memory for base components")

            available_for_layers = free_mem - base_mem
            max_layers = min(
                int(available_for_layers // per_layer_mem),
                config.num_hidden_layers
            )
            num_layers = max(max_layers, 1)
            print(f"Automatically selected {num_layers}/{config.num_hidden_layers} layers")
        except Exception as e:
            print(f"Layer adjustment failed: {e}. Using full model.")
            num_layers = config.num_hidden_layers
    else:
        num_layers = num_layers or config.num_hidden_layers

    # Load model with layer truncation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation='eager',
    )
    
    if num_layers < config.num_hidden_layers:
        model.model.layers = model.model.layers[:num_layers]
        model.config.num_hidden_layers = num_layers

    model.to(device)
    model.eval()
    return model, tokenizer, num_layers


class EmbedsAsInputWrapper(torch.nn.Module):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    def forward(self, inputs_embeds, index_select = None):
        with torch.inference_mode():
            out = self.llm.forward(input_ids = None,
                            attention_mask = None,
                            position_ids = None,
                            past_key_values = None,
                            inputs_embeds = inputs_embeds,
                            index_select = index_select)
        
        return out[0]

def generate_text(model, tokenizer, prompt, max_new_tokens=7):
    """Perform inference with proper resource management."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def inference(model_id='meta-llama/Llama-2-7b-chat-hf',device='cuda' ,num_layers=2):
    model, tokenizer, num_layers = get_hf_model(model_id, device, num_layers)
    prompt = "San Francisco is a"
    
    result = generate_text(model, tokenizer, prompt)
    print(f"\nnum_layers = {num_layers}, Generated text: {result}")
    # 2: totalitéaletoreignersbyMSран
    # 32: 

if __name__ == "__main__":
    
    fire.Fire(inference)

    