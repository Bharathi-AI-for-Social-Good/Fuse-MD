from typing import Dict, List, Tuple

import timm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .local_store import runtime_text_model_path


def resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def infer_hidden_size(llama: AutoModelForCausalLM) -> int:
    if hasattr(llama.config, "hidden_size"):
        return int(llama.config.hidden_size)
    if hasattr(llama.config, "n_embd"):
        return int(llama.config.n_embd)
    raise ValueError("Could not infer hidden size from the language model config")


class ImageEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = timm.create_model(model_name, pretrained=True, num_classes=embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.embedding(images)


class LlamaTextProjector(nn.Module):
    def __init__(self, llama: AutoModelForCausalLM, embedding_dim: int) -> None:
        super().__init__()
        self.text_base = getattr(llama, "model", llama)
        hidden_size = infer_hidden_size(llama)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 512, bias=False),
            nn.Dropout(0.5),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, embedding_dim, bias=False),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled.to(self.projection[0].weight.dtype)
        return self.projection(pooled)


class FusionHead(nn.Module):
    def __init__(self, fusion: str, embedding_dim: int) -> None:
        super().__init__()
        self.fusion = fusion

        if fusion == "concat":
            fused_dim = embedding_dim * 2
        elif fusion == "element":
            fused_dim = embedding_dim
        elif fusion == "avgpool":
            self.w_image = nn.Linear(embedding_dim, 64, bias=False)
            self.w_text = nn.Linear(embedding_dim, 64, bias=False)
            self.pool = nn.AvgPool1d(kernel_size=4)
            fused_dim = 16
        elif fusion == "gated":
            self.u_text = nn.Linear(embedding_dim, 64, bias=False)
            self.v_image = nn.Linear(embedding_dim, 64, bias=False)
            self.w_fused = nn.Linear(64, 8, bias=False)
            self.sigmoid = nn.Sigmoid()
            fused_dim = 8
        else:
            raise ValueError(f"Unsupported fusion method: {fusion}")

        hidden_dim = max(fused_dim // 2, 1)
        self.initial = nn.Linear(fused_dim, hidden_dim, bias=True)
        self.final = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        target_dtype = self.initial.weight.dtype
        text_features = text_features.to(target_dtype)
        image_features = image_features.to(target_dtype)

        if self.fusion == "concat":
            fused = torch.cat([text_features, image_features], dim=-1)
        elif self.fusion == "element":
            fused = text_features * image_features
        elif self.fusion == "avgpool":
            text_encoded = self.w_text(text_features)
            image_encoded = self.w_image(image_features)
            fused = self.pool((text_encoded * image_encoded).unsqueeze(1)).squeeze(1)
        else:
            gate = self.sigmoid(self.u_text(text_features))
            image_encoded = self.v_image(image_features)
            fused = self.w_fused(gate * image_encoded)

        hidden = self.initial(fused)
        return self.final(hidden).squeeze(-1)


class FuseMD(nn.Module):
    def __init__(
        self,
        text_encoder: LlamaTextProjector,
        image_encoder: ImageEmbeddingModel,
        fusion_head: FusionHead,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_head = fusion_head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(images)
        return self.fusion_head(text_features, image_features)


def load_tokenizer_and_llama(
    text_model_name: str,
    use_8bit: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    resolved_model_path = runtime_text_model_path(text_model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    quantization_config = None
    if use_8bit and device.type == "cuda":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_dtype = dtype if device.type == "cuda" else torch.float32
    llama = AutoModelForCausalLM.from_pretrained(
        resolved_model_path,
        low_cpu_mem_usage=True,
        local_files_only=True,
        quantization_config=quantization_config,
        torch_dtype=model_dtype,
    )
    if len(tokenizer) != llama.get_input_embeddings().num_embeddings:
        llama.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return tokenizer, llama


def freeze_base_models(model: FuseMD) -> None:
    for param in model.text_encoder.text_base.parameters():
        param.requires_grad = False
    for param in model.text_encoder.projection.parameters():
        param.requires_grad = True

    for param in model.image_encoder.embedding.parameters():
        param.requires_grad = False
    if hasattr(model.image_encoder.embedding, "head"):
        for param in model.image_encoder.embedding.head.parameters():
            param.requires_grad = True

    for param in model.fusion_head.parameters():
        param.requires_grad = True


def trainable_parameters(model: FuseMD) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def get_trainable_checkpoint(model: FuseMD) -> Dict[str, object]:
    vit_state = model.image_encoder.state_dict()
    vit_head_state = {key: value.cpu() for key, value in vit_state.items() if key.startswith("embedding.head")}

    return {
        "text_projection_state_dict": {
            key: value.cpu() for key, value in model.text_encoder.projection.state_dict().items()
        },
        "vit_head_state_dict": vit_head_state,
        "fusion_head_state_dict": {
            key: value.cpu() for key, value in model.fusion_head.state_dict().items()
        },
    }


def load_trainable_checkpoint(model: FuseMD, checkpoint: Dict[str, object]) -> None:
    text_projection_state = checkpoint.get("text_projection_state_dict")
    if text_projection_state is None:
        text_projection_state = checkpoint.get("text_clf_head_state_dict")
    if text_projection_state is None:
        raise KeyError("Checkpoint is missing both 'text_projection_state_dict' and 'text_clf_head_state_dict'.")

    model.text_encoder.projection.load_state_dict(text_projection_state)

    vit_current = model.image_encoder.state_dict()
    vit_current.update(checkpoint["vit_head_state_dict"])
    model.image_encoder.load_state_dict(vit_current, strict=False)

    model.fusion_head.load_state_dict(checkpoint["fusion_head_state_dict"])


def build_model(
    llama: AutoModelForCausalLM,
    image_model_name: str,
    fusion_method: str,
    embedding_dim: int,
) -> FuseMD:
    text_encoder = LlamaTextProjector(llama=llama, embedding_dim=embedding_dim)
    image_encoder = ImageEmbeddingModel(image_model_name, embedding_dim=embedding_dim)
    fusion_head = FusionHead(fusion=fusion_method, embedding_dim=embedding_dim)
    model = FuseMD(text_encoder=text_encoder, image_encoder=image_encoder, fusion_head=fusion_head)
    freeze_base_models(model)
    return model
