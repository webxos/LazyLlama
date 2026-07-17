"""
micro_moe.py - Micro Mixture of Experts for sub-1B student models.
Replaces dense FFN layers with sparse MoE layers during distillation or pruning.
Now supports static routing for deterministic expert assignment, improving
fusion with llama.cpp/Ollama and reducing variance.

Static routing uses KMeans clustering on calibration activations to pre‑assign
each token to a fixed expert, bypassing the learned router during inference.
This makes the model more deterministic and easier to fuse.

FIXES (2026-07-10):
- Added logging for better error visibility.
- Improved static router creation with fallback when scikit-learn is missing.
- Added early exit if calibration prompts are empty.
- Enhanced device handling and memory efficiency.
- Added type hints for better maintainability.
- Fixed Conv1D attribute error for GPT‑2 models (use nx/nf instead of in_features/out_features).

ENHANCEMENTS (2026-07-15):
- Expanded expert count up to 16+ and top_k up to 4.
- Added optional `capacity_factor` to limit tokens per expert.
- Implemented hierarchical routing: top‑level router selects a group, then second router picks within the group.
- `convert_dense_to_micro_moe` now accepts `hierarchical` parameter.
- `create_static_router` works with num_experts > 4.

FIXES (2026-07-15):
- Fixed auxiliary loss for hierarchical routing: now stores expert‑level logits (shape tokens x num_experts)
  instead of group logits, so `compute_auxiliary_loss` works correctly.
- Fixed critical output accumulation bug in hierarchical mode: now uses `mask_expert` instead of group `mask`
  when adding expert outputs, so only tokens assigned to that expert receive the contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
import gc
from typing import List, Optional, Tuple, Dict, Any, Union

# scikit-learn is required for static routing; fallback if not available
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None

logger = logging.getLogger(__name__)


def create_static_router(
    model: nn.Module,
    calibration_prompts: List[str],
    tokenizer,
    num_experts: int = 4,
    device: str = "cpu",
    max_samples: Optional[int] = None
) -> Optional[Any]:
    """
    Pre‑compute deterministic expert assignment from calibration activations.
    Returns a fitted KMeans object that can map embeddings to expert IDs.

    Args:
        model: The model (must have output_hidden_states=True in forward).
        calibration_prompts: List of strings for activation collection.
        tokenizer: Tokenizer for the model.
        num_experts: Number of clusters (experts). Works with any number > 0.
        device: Device to run on.
        max_samples: Maximum number of samples to use for clustering. If None, uses all.

    Returns:
        Fitted KMeans object, or None if scikit-learn is not available.

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If no activations are collected or calibration_prompts is empty.
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for static routing. "
            "Install it with: pip install scikit-learn"
        )

    if not calibration_prompts:
        raise ValueError("calibration_prompts cannot be empty for static router creation.")

    model.eval()
    activations = []
    device = torch.device(device)

    with torch.no_grad():
        for prompt in calibration_prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                outputs = model(**inputs, output_hidden_states=True)
                if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                    logger.warning("Model did not return hidden_states. Skipping static router creation.")
                    return None
                hidden = outputs.hidden_states[-1].mean(dim=1)  # (batch, hidden_dim)
                activations.append(hidden.cpu())
            except Exception as e:
                logger.warning(f"Failed to collect activations for prompt '{prompt[:30]}...': {e}")
                continue

    if not activations:
        raise ValueError("No activations collected from calibration prompts.")

    acts = torch.cat(activations, dim=0).numpy()  # (n_samples, hidden_dim)

    if max_samples is not None and acts.shape[0] > max_samples:
        indices = np.random.choice(acts.shape[0], max_samples, replace=False)
        acts = acts[indices]

    # Ensure we have at least num_experts samples
    if acts.shape[0] < num_experts:
        repeats = (num_experts // acts.shape[0]) + 1
        acts = np.repeat(acts, repeats, axis=0)[:num_experts]
        logger.warning(
            f"Only {len(activations)} samples collected, which is less than num_experts={num_experts}. "
            "Duplicating samples for clustering."
        )

    try:
        kmeans = KMeans(n_clusters=num_experts, random_state=42, n_init=10).fit(acts)
        logger.info(f"Static router created with {num_experts} clusters.")
        return kmeans
    except Exception as e:
        logger.error(f"Failed to fit KMeans: {e}")
        return None


class Sub1BExpert(nn.Module):
    """A small FFN expert with SiLU activation."""
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.act(self.w1(x)))


class MicroMoELayer(nn.Module):
    """
    Micro MoE layer with top-k routing.
    Stores router_logits during forward for auxiliary loss computation.
    Supports static routing via a pre‑computed KMeans router.
    Now supports up to 16+ experts, top-k up to 4, capacity factor, and hierarchical routing.
    """
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        use_static_routing: bool = False,
        static_router=None,
        capacity_factor: Optional[float] = None,
        hierarchical: bool = False,
        groups: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # ensure top_k <= num_experts
        self.use_static_routing = use_static_routing
        self.static_router = static_router
        self.capacity_factor = capacity_factor
        self.hierarchical = hierarchical

        if self.hierarchical:
            # Determine groups: default to 2 groups if not specified
            if groups is None:
                groups = 2
            self.groups = min(groups, num_experts)
            self.experts_per_group = num_experts // self.groups
            if self.experts_per_group < 1:
                raise ValueError(f"Groups {self.groups} too large for {num_experts} experts.")
            # Top-level router: predicts group index
            self.group_router = nn.Linear(hidden_dim, self.groups, bias=False)
            # Per-group routers: each is a linear layer from hidden_dim to experts_per_group
            self.group_routers = nn.ModuleList([
                nn.Linear(hidden_dim, self.experts_per_group, bias=False)
                for _ in range(self.groups)
            ])
            # Experts are stored in a flat list, grouped by group
            self.experts = nn.ModuleList([
                Sub1BExpert(hidden_dim, intermediate_dim) for _ in range(num_experts)
            ])
            self._last_router_logits = None  # will store combined logits for auxiliary loss
        else:
            self.router = nn.Linear(hidden_dim, num_experts, bias=False)
            self.experts = nn.ModuleList([
                Sub1BExpert(hidden_dim, intermediate_dim) for _ in range(num_experts)
            ])
            self.group_router = None
            self.group_routers = None

        # Initialize storage for router logits (used by distillation helpers)
        self._last_router_logits = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_dim)
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])  # (tokens, hidden_dim)
        num_tokens = x_flat.size(0)

        if self.use_static_routing and self.static_router is not None:
            # Deterministic assignment: aggregate per batch item
            batch_acts = x.mean(dim=1)  # (batch, hidden_dim)
            try:
                expert_ids = self.static_router.predict(batch_acts.cpu().numpy())  # (batch,)
            except Exception as e:
                logger.error(f"Static routing prediction failed: {e}")
                return self._dynamic_forward(x_flat, orig_shape, num_tokens)

            expert_ids = torch.tensor(expert_ids, device=x.device).long()
            token_expert_ids = expert_ids.unsqueeze(1).expand(-1, x.size(1)).reshape(-1)  # (batch*seq_len,)
            output = torch.zeros_like(x_flat)
            for i, expert in enumerate(self.experts):
                mask = (token_expert_ids == i).unsqueeze(-1)
                if mask.any():
                    tokens = x_flat[mask.squeeze(-1)]
                    expert_out = expert(tokens)
                    output[mask.squeeze(-1)] = expert_out
            self._last_router_logits = None
            return output.view(orig_shape)

        # Dynamic routing
        return self._dynamic_forward(x_flat, orig_shape, num_tokens)

    def _dynamic_forward(self, x_flat: torch.Tensor, orig_shape: torch.Size, num_tokens: int) -> torch.Tensor:
        """
        Dynamic routing implementation. Handles hierarchical routing, capacity factor, and top-k.
        Returns the output tensor and updates self._last_router_logits with expert-level logits.
        """
        if self.hierarchical:
            # Hierarchical routing:
            # 1. Compute group logits
            group_logits = self.group_router(x_flat)  # (tokens, groups)
            group_probs = F.softmax(group_logits, dim=-1)
            # 2. Hard assignment to the most likely group
            group_indices = torch.argmax(group_probs, dim=-1)  # (tokens,)

            # Prepare output tensor
            output = torch.zeros_like(x_flat)

            # We need to build expert-level logits for auxiliary loss.
            expert_logits = torch.full((num_tokens, self.num_experts), -1e9, device=x_flat.device)

            # For each group, process tokens assigned to that group
            for g in range(self.groups):
                mask = (group_indices == g)
                if not mask.any():
                    continue
                tokens_g = x_flat[mask]  # (n_g, hidden_dim)
                n_g = tokens_g.size(0)

                # Group-specific router
                group_router = self.group_routers[g]
                logits_g = group_router(tokens_g)  # (n_g, experts_per_group)
                # Apply top-k within the group
                routing_weights = F.softmax(logits_g, dim=-1)
                topk_weights, topk_indices = torch.topk(
                    routing_weights, min(self.top_k, self.experts_per_group), dim=-1
                )
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

                # Compute expert outputs
                global_base = g * self.experts_per_group

                # For each token in this group, we have selected topk_indices and topk_weights
                # We'll loop over top_k positions to accumulate contributions
                for k in range(topk_indices.size(1)):
                    local_expert_idx = topk_indices[:, k]  # (n_g,)
                    weight = topk_weights[:, k].unsqueeze(-1)  # (n_g, 1)
                    global_expert_idx = global_base + local_expert_idx

                    # Process per unique expert to avoid repeated expert calls
                    unique_experts = torch.unique(global_expert_idx)
                    for expert_idx in unique_experts:
                        mask_expert = (global_expert_idx == expert_idx)
                        if not mask_expert.any():
                            continue
                        tokens_expert = tokens_g[mask_expert]
                        weight_expert = weight[mask_expert]
                        expert = self.experts[expert_idx.item()]
                        expert_out = expert(tokens_expert)
                        # FIX: use mask_expert, not the group mask, to add only to assigned tokens
                        output[mask_expert] += weight_expert * expert_out

                # Fill expert_logits for auxiliary loss
                # For tokens in this group, expert logit = group_logit[g] + logits_g[expert]
                group_logit_token = group_logits[mask, g].unsqueeze(1)  # (n_g,1)
                # We'll assign for each token and each expert in the group
                token_indices = torch.nonzero(mask).squeeze(-1)  # (n_g,)
                for i, token_idx in enumerate(token_indices.tolist()):
                    expert_logits[token_idx, global_base:global_base+self.experts_per_group] = \
                        group_logit_token[i] + logits_g[i]

            # Store expert-level logits for auxiliary loss
            self._last_router_logits = expert_logits

            output = output.view(orig_shape)
            return output

        else:
            # Standard non-hierarchical routing
            router_logits = self.router(x_flat)                 # (tokens, num_experts)
            routing_weights = F.softmax(router_logits, dim=-1)

            # Optional capacity factor: limit tokens per expert
            if self.capacity_factor is not None:
                # Compute token count per expert
                topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

                flat_indices = topk_indices.view(-1)
                expert_counts = torch.bincount(flat_indices, minlength=self.num_experts)
                capacity = int((num_tokens / self.num_experts) * self.capacity_factor)
                if capacity < 1:
                    capacity = 1

                # For each expert, keep only top `capacity` tokens by weight
                tokens_per_expert = [[] for _ in range(self.num_experts)]
                for i in range(num_tokens):
                    for k in range(self.top_k):
                        expert_idx = topk_indices[i, k].item()
                        weight = topk_weights[i, k].item()
                        tokens_per_expert[expert_idx].append((i, weight))

                output = torch.zeros_like(x_flat)
                for expert_idx, token_list in enumerate(tokens_per_expert):
                    if not token_list:
                        continue
                    # Sort by weight descending
                    token_list.sort(key=lambda x: x[1], reverse=True)
                    selected = token_list[:capacity]
                    if not selected:
                        continue
                    selected_indices = [t[0] for t in selected]
                    selected_weights = torch.tensor([t[1] for t in selected], device=x_flat.device)
                    tokens_selected = x_flat[selected_indices]
                    expert = self.experts[expert_idx]
                    expert_out = expert(tokens_selected)
                    output[selected_indices] += selected_weights.unsqueeze(-1) * expert_out

                # For tokens that were not assigned to any expert (due to capacity), fallback to expert 0
                assigned = (output != 0).any(dim=-1)
                fallback_mask = ~assigned
                if fallback_mask.any():
                    fallback_tokens = x_flat[fallback_mask]
                    expert0_out = self.experts[0](fallback_tokens)
                    output[fallback_mask] = expert0_out

                self._last_router_logits = router_logits
                output = output.view(orig_shape)
                return output

            else:
                # No capacity factor: standard top-k
                topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
                topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

                output = torch.zeros_like(x_flat)

                # Loop over experts
                for i, expert in enumerate(self.experts):
                    # For each expert, find which tokens selected it
                    token_mask = (topk_indices == i).any(dim=-1)  # (tokens,)
                    if not token_mask.any():
                        continue
                    # Get the tokens
                    tokens = x_flat[token_mask]
                    # Get the weights for this expert for each token
                    # We'll use a loop over k to accumulate.
                    for k in range(self.top_k):
                        pos_mask = (topk_indices[:, k] == i)  # (tokens,)
                        if not pos_mask.any():
                            continue
                        weight = topk_weights[pos_mask, k]  # (n_pos,)
                        tokens_k = x_flat[pos_mask]
                        expert_out = expert(tokens_k)
                        output[pos_mask] += weight.unsqueeze(-1) * expert_out

                self._last_router_logits = router_logits
                output = output.view(orig_shape)
                return output


def _is_ffn_module(module: nn.Module) -> bool:
    """Heuristic to identify FFN modules in common architectures."""
    if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
        return True
    if hasattr(module, 'intermediate') and hasattr(module, 'output'):
        return True
    if hasattr(module, 'c_fc') and hasattr(module, 'c_proj'):
        return True
    return False


def _get_ffn_params(module: nn.Module, hidden_dim: int) -> Optional[Tuple[int, nn.Module]]:
    """
    Extract intermediate dimension and the module itself if it's an FFN.
    Returns (intermediate_dim, module) or None.
    """
    if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
        inter_dim = module.gate_proj.out_features
        return inter_dim, module
    if hasattr(module, 'intermediate'):
        if hasattr(module.intermediate, 'dense'):
            inter_dim = module.intermediate.dense.out_features
            return inter_dim, module
    if hasattr(module, 'c_fc'):
        inter_dim = module.c_fc.nf
        return inter_dim, module
    return None


def convert_dense_to_micro_moe(
    model: nn.Module,
    num_experts: int = 4,
    top_k: int = 1,
    reduction_factor: int = 2,
    init_noise_std: float = 1e-3,
    use_static_routing: bool = False,
    static_router=None,
    copy_model: bool = True,
    capacity_factor: Optional[float] = None,
    hierarchical: bool = False,
    groups: Optional[int] = None,
) -> nn.Module:
    """
    Recursively replace FFN modules with MicroMoELayer.
    The intermediate dimension per expert is reduced by reduction_factor.
    Initializes experts from the original dense layer weights (splitted with noise).

    Args:
        model: The model to convert.
        num_experts: Number of experts per MoE layer. Can be up to 16+.
        top_k: Top-k routing (only used for dynamic routing; static routing uses deterministic assignment).
               Max top_k is 4, but can be larger.
        reduction_factor: Reduce intermediate size per expert by this factor.
        init_noise_std: Standard deviation of noise added to initial weights.
        use_static_routing: If True, use deterministic static routing instead of learned router.
        static_router: Fitted KMeans object (required if use_static_routing=True). If None,
                       and use_static_routing=True, a ValueError is raised.
        copy_model: If True, create a deep copy of the model before modification (safe). Default True.
        capacity_factor: If provided, limits tokens per expert to capacity_factor * average tokens per expert.
                         Currently only applied in non-hierarchical mode.
        hierarchical: If True, use hierarchical routing (top-level groups then within-group).
        groups: Number of groups for hierarchical routing. If None, defaults to 2.

    Returns:
        The modified model (either original or a copy).
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if top_k > num_experts:
        logger.warning(f"top_k ({top_k}) > num_experts ({num_experts}); setting top_k = num_experts.")
        top_k = num_experts
    if top_k < 1:
        top_k = 1

    if use_static_routing and static_router is None:
        raise ValueError("static_router must be provided when use_static_routing=True")

    if copy_model:
        model = copy.deepcopy(model)

    # First, find all FFN modules and their paths
    replacements = []
    for name, module in model.named_modules():
        if _is_ffn_module(module):
            # Determine hidden_dim and intermediate_dim
            if hasattr(module, 'gate_proj'):
                hidden_dim = module.gate_proj.in_features
                intermediate_dim = module.gate_proj.out_features
            elif hasattr(module, 'intermediate') and hasattr(module.intermediate, 'dense'):
                hidden_dim = module.intermediate.dense.in_features
                intermediate_dim = module.intermediate.dense.out_features
            elif hasattr(module, 'c_fc'):
                hidden_dim = module.c_fc.nx
                intermediate_dim = module.c_fc.nf
            else:
                continue

            # Build MoE layer with reduced intermediate size per expert
            expert_inter_dim = max(1, intermediate_dim // reduction_factor)
            moe_layer = MicroMoELayer(
                hidden_dim=hidden_dim,
                intermediate_dim=expert_inter_dim,
                num_experts=num_experts,
                top_k=top_k,
                use_static_routing=use_static_routing,
                static_router=static_router,
                capacity_factor=capacity_factor,
                hierarchical=hierarchical,
                groups=groups,
            )
            replacements.append((name, module, moe_layer, hidden_dim, intermediate_dim, expert_inter_dim))

    if not replacements:
        logger.warning("No FFN modules found in the model. Conversion may have no effect.")
        return model

    # Perform replacements
    logger.info(f"Found {len(replacements)} FFN modules to convert to MoE")
    for name, old_module, moe_layer, hidden_dim, intermediate_dim, expert_inter_dim in replacements:
        # Split the original weight matrices across experts
        if hasattr(old_module, 'gate_proj'):
            gate_weight = old_module.gate_proj.weight.data
            up_weight = old_module.up_proj.weight.data
            down_weight = old_module.down_proj.weight.data

            chunk_size = max(1, intermediate_dim // num_experts)
            for i, expert in enumerate(moe_layer.experts):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, intermediate_dim)
                rows_available = end - start
                rows_to_take = min(expert_inter_dim, rows_available)
                if rows_to_take == 0:
                    w1_slice = torch.randn(expert_inter_dim, hidden_dim) * init_noise_std
                else:
                    w1_slice = gate_weight[start:start+rows_to_take, :]
                    if rows_to_take < expert_inter_dim:
                        pad_rows = expert_inter_dim - rows_to_take
                        w1_slice = torch.cat([w1_slice, torch.randn(pad_rows, hidden_dim) * init_noise_std], dim=0)
                w1_slice += torch.randn_like(w1_slice) * init_noise_std
                expert.w1.weight.data = w1_slice.clone()

                rows_available = end - start
                rows_to_take = min(expert_inter_dim, rows_available)
                if rows_to_take == 0:
                    w2_slice = torch.randn(hidden_dim, expert_inter_dim) * init_noise_std
                else:
                    w2_slice = down_weight[:, start:start+rows_to_take]
                    if rows_to_take < expert_inter_dim:
                        pad_cols = expert_inter_dim - rows_to_take
                        w2_slice = torch.cat([w2_slice, torch.randn(hidden_dim, pad_cols) * init_noise_std], dim=1)
                w2_slice += torch.randn_like(w2_slice) * init_noise_std
                expert.w2.weight.data = w2_slice.clone()

        elif hasattr(old_module, 'intermediate') and hasattr(old_module.intermediate, 'dense'):
            inter_weight = old_module.intermediate.dense.weight.data
            out_weight = old_module.output.dense.weight.data
            chunk_size = max(1, intermediate_dim // num_experts)
            for i, expert in enumerate(moe_layer.experts):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, intermediate_dim)
                rows_available = end - start
                rows_to_take = min(expert_inter_dim, rows_available)
                if rows_to_take == 0:
                    w1_slice = torch.randn(expert_inter_dim, hidden_dim) * init_noise_std
                else:
                    w1_slice = inter_weight[start:start+rows_to_take, :]
                    if rows_to_take < expert_inter_dim:
                        pad_rows = expert_inter_dim - rows_to_take
                        w1_slice = torch.cat([w1_slice, torch.randn(pad_rows, hidden_dim) * init_noise_std], dim=0)
                w1_slice += torch.randn_like(w1_slice) * init_noise_std
                expert.w1.weight.data = w1_slice.clone()

                rows_available = end - start
                rows_to_take = min(expert_inter_dim, rows_available)
                if rows_to_take == 0:
                    w2_slice = torch.randn(hidden_dim, expert_inter_dim) * init_noise_std
                else:
                    w2_slice = out_weight[:, start:start+rows_to_take]
                    if rows_to_take < expert_inter_dim:
                        pad_cols = expert_inter_dim - rows_to_take
                        w2_slice = torch.cat([w2_slice, torch.randn(hidden_dim, pad_cols) * init_noise_std], dim=1)
                w2_slice += torch.randn_like(w2_slice) * init_noise_std
                expert.w2.weight.data = w2_slice.clone()

        elif hasattr(old_module, 'c_fc'):
            fc_weight = old_module.c_fc.weight.data
            proj_weight = old_module.c_proj.weight.data
            chunk_size = max(1, intermediate_dim // num_experts)
            for i, expert in enumerate(moe_layer.experts):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, intermediate_dim)
                rows_available = end - start
                rows_to_take = min(expert_inter_dim, rows_available)
                if rows_to_take == 0:
                    w1_slice = torch.randn(expert_inter_dim, hidden_dim) * init_noise_std
                else:
                    w1_slice = fc_weight[start:start+rows_to_take, :]
                    if rows_to_take < expert_inter_dim:
                        pad_rows = expert_inter_dim - rows_to_take
                        w1_slice = torch.cat([w1_slice, torch.randn(pad_rows, hidden_dim) * init_noise_std], dim=0)
                w1_slice += torch.randn_like(w1_slice) * init_noise_std
                expert.w1.weight.data = w1_slice.clone()

                rows_available = end - start
                rows_to_take = min(expert_inter_dim, rows_available)
                if rows_to_take == 0:
                    w2_slice = torch.randn(hidden_dim, expert_inter_dim) * init_noise_std
                else:
                    w2_slice = proj_weight[:, start:start+rows_to_take]
                    if rows_to_take < expert_inter_dim:
                        pad_cols = expert_inter_dim - rows_to_take
                        w2_slice = torch.cat([w2_slice, torch.randn(hidden_dim, pad_cols) * init_noise_std], dim=1)
                w2_slice += torch.randn_like(w2_slice) * init_noise_std
                expert.w2.weight.data = w2_slice.clone()

        # Replace the module in the parent
        parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else ''
        child_name = name.split('.')[-1]
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        setattr(parent, child_name, moe_layer)
        logger.info(f"Replaced FFN module {name} with MicroMoELayer (experts={num_experts}, top_k={top_k})")

    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return model


def compute_auxiliary_loss(router_logits_list: List[torch.Tensor], num_experts: int) -> torch.Tensor:
    """
    Auxiliary load-balancing loss to encourage uniform expert usage.
    router_logits_list: list of tensors of shape (tokens, num_experts) from each MoE layer.
    Returns a scalar loss tensor.

    This expects expert-level logits of shape (tokens, num_experts). For hierarchical routing,
    the logits should be combined as described in the MicroMoELayer implementation.
    """
    total_loss = 0.0
    for logits in router_logits_list:
        if logits.size(0) == 0:
            continue
        # Ensure logits has correct shape (tokens, num_experts)
        if logits.size(-1) != num_experts:
            logger.warning(
                f"Expected router logits shape (tokens, {num_experts}), got {logits.shape}. "
                "Skipping this layer in auxiliary loss."
            )
            continue
        probs = F.softmax(logits, dim=-1)
        token_choices = torch.argmax(probs, dim=-1)
        density = torch.bincount(token_choices, minlength=num_experts).float() / logits.size(0)
        avg_probs = probs.mean(dim=0)
        total_loss += num_experts * torch.sum(density * avg_probs)
    return total_loss