import torch
import torch.nn as nn
from transformers import (
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel
)


class UNetModernBERT(nn.Module):
    """
    U-Net style architecture with:
    - Convolutional downsampling (encoder)
    - ModernBERT in the middle (bottleneck)
    - Convolutional upsampling (decoder)
    - Skip connections between encoder and decoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        self.gradient_checkpointing = False

        # Embedding layer
        self.embeddings = nn.Embedding(config.vocab_size, hidden_size, padding_idx=config.pad_token_id)

        # Encoder: Downsampling with 1D convolutions
        # We'll use 2 downsampling stages to reduce sequence length by 4x
        self.encoder_conv1 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.encoder_conv2 = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        # ModernBERT bottleneck
        # Adjust max_position_embeddings for the bottleneck (downsampled by 4x)
        bottleneck_config = ModernBertConfig(
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings // 4,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            hidden_size=hidden_size,
            intermediate_size=config.intermediate_size,
            type_vocab_size=config.type_vocab_size,
            hidden_activation=config.hidden_activation,
            global_attn_every_n_layers=config.global_attn_every_n_layers,
            local_attention=config.local_attention // 4,  # Adjust local attention window
            deterministic_flash_attn=config.deterministic_flash_attn,
            global_rope_theta=config.global_rope_theta,
            local_rope_theta=config.local_rope_theta,
            pad_token_id=config.pad_token_id,
        )
        self.modernbert = ModernBertModel(bottleneck_config)

        # Decoder: Upsampling with transposed convolutions
        self.decoder_conv1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size * 2, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        # Language modeling head
        self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the ModernBERT bottleneck."""
        self.gradient_checkpointing = True
        if hasattr(self.modernbert, 'gradient_checkpointing_enable'):
            self.modernbert.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if hasattr(self.modernbert, 'gradient_checkpointing_disable'):
            self.modernbert.gradient_checkpointing_disable()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape

        # Embed input
        x = self.embeddings(input_ids)  # [batch, seq_len, hidden_size]

        # Transpose for Conv1d (expects [batch, channels, length])
        x = x.transpose(1, 2)  # [batch, hidden_size, seq_len]

        # Encoder stage 1 with skip connection
        skip1 = x
        x = self.encoder_conv1[0](x)  # Conv
        x = x.transpose(1, 2)  # [batch, seq_len/2, hidden_size]
        x = self.encoder_conv1[2](x)  # LayerNorm
        x = self.encoder_conv1[1](x)  # GELU
        x = x.transpose(1, 2)  # [batch, hidden_size, seq_len/2]

        # Encoder stage 2 with skip connection
        skip2 = x
        x = self.encoder_conv2[0](x)  # Conv
        x = x.transpose(1, 2)  # [batch, seq_len/4, hidden_size]
        x = self.encoder_conv2[2](x)  # LayerNorm
        x = self.encoder_conv2[1](x)  # GELU
        # x is now [batch, seq_len/4, hidden_size]

        # Create attention mask for downsampled sequence
        if attention_mask is not None:
            # Downsample attention mask by max pooling
            downsampled_mask = attention_mask.unsqueeze(1).float()  # [batch, 1, seq_len]
            downsampled_mask = nn.functional.max_pool1d(downsampled_mask, kernel_size=2, stride=2)
            downsampled_mask = nn.functional.max_pool1d(downsampled_mask, kernel_size=2, stride=2)
            downsampled_mask = downsampled_mask.squeeze(1).long()  # [batch, seq_len/4]
        else:
            downsampled_mask = None

        # ModernBERT bottleneck
        bert_output = self.modernbert(
            inputs_embeds=x,
            attention_mask=downsampled_mask,
            return_dict=True
        )
        x = bert_output.last_hidden_state  # [batch, seq_len/4, hidden_size]

        # Decoder stage 1 with skip connection
        x = x.transpose(1, 2)  # [batch, hidden_size, seq_len/4]
        x = torch.cat([x, skip2], dim=1)  # [batch, hidden_size*2, seq_len/4]
        x = self.decoder_conv1[0](x)  # ConvTranspose
        x = x.transpose(1, 2)  # [batch, seq_len/2, hidden_size]
        x = self.decoder_conv1[2](x)  # LayerNorm
        x = self.decoder_conv1[1](x)  # GELU
        x = x.transpose(1, 2)  # [batch, hidden_size, seq_len/2]

        # Decoder stage 2 with skip connection
        # Ensure skip1 has the right size for concatenation
        if skip1.size(2) != x.size(2):
            skip1 = skip1[:, :, :x.size(2)]
        x = torch.cat([x, skip1], dim=1)  # [batch, hidden_size*2, seq_len/2]
        x = self.decoder_conv2[0](x)  # ConvTranspose
        x = x.transpose(1, 2)  # [batch, seq_len, hidden_size]
        x = self.decoder_conv2[2](x)  # LayerNorm
        x = self.decoder_conv2[1](x)  # GELU

        # Ensure output sequence length matches input
        if x.size(1) != seq_len:
            x = x[:, :seq_len, :]

        # Language modeling head
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        # Return in the format expected by HuggingFace
        return type('ModelOutput', (), {
            'logits': logits,
            'last_hidden_state': x,
            'hidden_states': None,
            'attentions': None,
        })()


class ProteinBertModel:
    def __init__(
        self,
        vocab_size,
        tokenizer,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=8192,
        local_attention=512,
    ):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.local_attention = local_attention

    def build(self):
        config = ModernBertConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            type_vocab_size=1,
            hidden_activation="gelu",
            global_attn_every_n_layers=3,
            local_attention=self.local_attention,
            deterministic_flash_attn=False,
            global_rope_theta=160000.0,
            local_rope_theta=10000.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            cls_token_id=self.tokenizer.cls_token_id,
            sep_token_id=self.tokenizer.sep_token_id,
        )
        # Use U-Net architecture with ModernBERT in the middle
        model = UNetModernBERT(config)
        return model