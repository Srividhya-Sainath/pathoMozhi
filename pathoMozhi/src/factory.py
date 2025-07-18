from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance

# Yeh jo function hai, yeh vision+language+adapter model ko banata hai. Hamare case mein vision element nahi hai kyunki
# humne directly vision ke features ko load kiya hai.
# Yeh function ek Tokenizer ko bhi initialize karta hai, jo ki language model ke liye hota hai.
def create_model_and_transforms(
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    cache_dir: Optional[str] = None,
    cls_type="both",
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
        freeze_lm_embeddings (bool, optional): whether to freeze LM input embeddings when configuring Perceiver.
        cache_dir (str, optional): path to cache directory for downloading OpenClip/HF weights.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_fast=True
    )

    SPECIAL_TASK_TOKENS = ["<image>", "<|endofchunk|>"]
    if cls_type in ["organ", "both"]:
        SPECIAL_TASK_TOKENS.append("<cls1>")
    if cls_type in ["diagnosis", "both"]:
        SPECIAL_TASK_TOKENS.append("<cls2>")

    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TASK_TOKENS}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # AutoModelForCausalLM jo hai, yeh logits deta hai next token prediction ke liye --> TODO: Isko ache se samaj !
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_safetensors=True
    )

    # hacks for MPT-1B, which doesn't have a get_input_embeddings method
    if "mpt-1b-redpajama-200b" in lang_encoder_path:

        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings

        extend_instance(lang_encoder, EmbeddingFnMixin)

    # convert LM to FlamingoLM
    # STEP2 --> Yeh samajna hai
    extend_instance(lang_encoder, FlamingoLMMixin)
    # Yahaan hum dekhte hain ki decoder blocks kahaan hain, aur uske hisaab se hum wahaan cross-attention layers add karte hain
    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer)) ## After adding special tokens (<PAD>, <|endofchunk|>, <image>), the tokenizer’s vocabulary size increases, and the embedding layer must be resized accordingly.

    # Yahin peh poora Flamingo model ban raha hai
    # STEP3 --> Yeh samajna hai
    model = Flamingo(
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim= 768,
        tokenizer=text_tokenizer,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        cls_type=cls_type,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    # Pehle sab kuch freeze karte hain
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Phir kuch layers ko unfreeze karte hain
    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        # TODO: investigate also training the output embeddings when untied


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Flamingo model initialized with {trainable_params} trainable parameters")
    print(f"Total trainable parameters: {trainable_params/1e6:.2f}M") 

    return model,text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "biogptforcausallm": "biogpt.layers",
}