import torch
from torch import nn
from .helpers import PerceiverResampler

class Flamingo(nn.Module):
    def __init__(
        self,
        lang_encoder: nn.Module,
        eoc_token_id: int, # The token ID for <|endofchunk|>
        media_token_id: int, # The token ID for <image>
        vis_dim: int,
        tokenizer,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
        num_organ_classes: int = 9,
        num_diagnosis_classes: int = 38,
        cls_type="both",
    ):
        """
        Args:
            lang_encoder (nn.Module): HF causal language model
            eoc_token_id (int): Token id for <|endofchunk|>
            media_token_id (int): Token id for <image>
            vis_dim (int): Dimension of the visual features.
                Visual features are projected to match this shape along the last dimension.
            cross_attn_every_n_layers (int, optional): How often to apply cross attention after transformer layer. Defaults to 1.
        """
        super().__init__()
        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.vis_dim = vis_dim # Ideally 768 hona chahiye. Both TITAN and CONCHv1.5 ka 768 hi hai
        self.tokenizer = tokenizer
        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            #self.lang_dim = lang_encoder.config.hidden_size
            self.lang_dim = lang_encoder.get_input_embeddings().embedding_dim
        self.cls_type = cls_type
        if cls_type in ["organ", "both"]:
            self.cls_head1 = nn.Linear(self.lang_dim, num_organ_classes)
        if cls_type in ["diagnosis", "both", "diagnosisnoclass"]:
            self.cls_head2 = nn.Linear(self.lang_dim, num_diagnosis_classes)
        if cls_type == "diagnosisAttn":
            self.attn_pool = nn.Linear(self.lang_dim, 1)
            self.cls_head2 = nn.Linear(self.lang_dim, num_diagnosis_classes)
        self.perceiver = PerceiverResampler(dim=self.vis_dim)
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_dim,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )
        self._use_gradient_checkpointing = gradient_checkpointing
        self.perceiver._use_gradient_checkpointing = gradient_checkpointing

    def forward(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """
        Forward pass of Flamingo.
        Args:
            vision_x (torch.Tensor): Precomputed image embeddings of shape [B, T_img, D]
            lang_x (torch.Tensor): Language input ids [B, T_txt]
            attention_mask (torch.Tensor, optional): Attention mask [B, T_txt]
            labels (torch.Tensor, optional): Labels for AR loss
            clear_conditioned_layers (bool): Whether to clear vision/text condition after pass
            past_key_values: For cached decoding (HuggingFace compat)
            use_cache (bool): Whether to use key/value cache (HuggingFace compat)
        Returns:
            Tuple: (ar_loss)
        """
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_vision_x or vision_x is not None
        ), "Must provide either vision_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)
            self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Optional: get final hidden states for classifier heads
        if output.hidden_states is not None:
            hidden_states = output.hidden_states[-1]  # (B, T, D)
            B = hidden_states.size(0)

            # Organ classification
            if self.cls_type in ["organ", "both"]:
                cls1_id = self.tokenizer.convert_tokens_to_ids("<cls1>")
                cls1_mask = (lang_x == cls1_id)
                if cls1_mask.sum(dim=1).min() < 1:
                    raise ValueError("Expected <cls1> token per sample for organ classification.")
                cls_feats1 = hidden_states[cls1_mask].view(B, -1)
                output["cls_logits1"] = self.cls_head1(cls_feats1)

            # Diagnosis classification (various strategies)
            if self.cls_type == "diagnosisnoclass":
                # Use last token embedding
                cls_feats2 = hidden_states[:, -1, :]  # (B, D)
                output["cls_logits2"] = self.cls_head2(cls_feats2)

            elif self.cls_type in ["diagnosis", "both"]:
                # Use <cls2> token representation
                cls2_id = self.tokenizer.convert_tokens_to_ids("<cls2>")
                cls2_mask = (lang_x == cls2_id)
                if cls2_mask.sum(dim=1).min() < 1:
                    raise ValueError("Expected <cls2> token per sample for diagnosis classification.")
                cls_feats2 = hidden_states[cls2_mask].view(B, -1)
                output["cls_logits2"] = self.cls_head2(cls_feats2)

            elif self.cls_type == "diagnosisAttn":
                # Use attention pooling over hidden states
                attn_weights = self.attn_pool(hidden_states).softmax(dim=1)  # (B, T, 1)
                pooled = (hidden_states * attn_weights).sum(dim=1)           # (B, D)
                output["cls_logits2"] = self.cls_head2(pooled)
        else:
            print("output.hidden_states is None")

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, D)
                B: Batch size, D: Embedding dimension
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            **kwargs: see generate documentation in Hugging Face CausalLM models. Some notable kwargs:
                max_length (int, optional): Maximum length of the output. Defaults to None.
                attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
                num_beams (int, optional): Number of beams. Defaults to 1.
                max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
                temperature (float, optional): Temperature. Defaults to 1.0.
                top_k (int, optional): Top k. Defaults to 50.
                top_p (float, optional): Top p. Defaults to 1.0.
                no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
                length_penalty (float, optional): Length penalty. Defaults to 1.0.
                num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
                do_sample (bool, optional): Do sample. Defaults to False.
                early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder.cached_input_ids = lang_x # Cache text tokens for `_encode_vision_x()`

        self.lang_encoder._use_cached_vision_x = True
        self._encode_vision_x(vision_x=vision_x)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            **kwargs,
            )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False
        return output
    
    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Prepare global embeddings to align with (B, T_img, F, V, D) format.
        Args:
            vision_x (torch.Tensor): Precomputed image embeddings of shape (B, T_img, D)
        Returns:
            vision_latents (torch.Tensor): Latent features to condition the LM
        """
        assert vision_x.ndim == 4, f"Expected image shape [B,Tm,N,D], got {vision_x.shape}"
        B, T_img, V, D = vision_x.shape

        # Reshape to match (B, T_img, F=1, V=1, D)
        vision_x = vision_x.view(B, T_img, 1, V, D)

        vision_x = self.perceiver(vision_x)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    def _condition_media_locations(self, input_ids: torch.Tensor):
        """
        Compute the media token locations from lang_x and condition the language model on these.
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
        """
        media_locations = input_ids == self.media_token_id

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: torch.Tensor, vision_x: torch.Tensor):
        """
        Pre-cache a prompt/sequence of images / text for log-likelihood evaluations.
        All subsequent calls to forward() will generate attending to the LAST
        image in vision_x.
        This is not meant to be used to cache things for generate().
        Args:
            input_ids (torch.Tensor): Language input
                shape (B, T_txt)
            vision_x (torch.Tensor): Precomputed image embeddings of shape (B, D).
                B: Batch size, D: Embedding dimension.
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos). NOTE: T_img is also 1 in our case
        """
        self._encode_vision_x(vision_x=vision_x)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_vision_x = True

    def uncache_media(self):
        """
        Clear all conditioning.
        """
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False