from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from training.models import GoldRushPolicyNetwork

from .dataset import BcBatch


@dataclass(frozen=True)
class BcLossConfig:
    w_ko: float = 1.0
    w_action: float = 1.0
    w_vp: float = 0.2


@dataclass(frozen=True)
class BcLossResult:
    loss: Tensor
    nll_ko: Tensor
    nll_action: Tensor
    nll_vp: Tensor
    accuracy_ko: Tensor
    accuracy_vp: Tensor


def bc_loss(model: GoldRushPolicyNetwork, batch: BcBatch, config: BcLossConfig | None = None) -> BcLossResult:
    config = BcLossConfig() if config is None else config
    evaluation = model.evaluate_bc_actions(batch.planes, batch.scalars, batch.actions, batch.k, batch.order, batch.vp)
    nll_ko = -evaluation.ko_logprob.mean()
    nll_action = -(evaluation.action_logprob / 6.0).mean()
    nll_vp = -evaluation.vp_logprob.mean()
    loss = config.w_ko * nll_ko + config.w_action * nll_action + config.w_vp * nll_vp

    with torch.no_grad():
        encoded = model._encode(batch.planes, batch.scalars)
        ko_logits = model.ko_head(encoded.actor_context)
        vp_logits = model.vp_head(encoded.actor_context)
        target_ko = 2 * batch.k.long() + batch.order.long()
        accuracy_ko = (ko_logits.argmax(dim=1) == target_ko).float().mean()
        accuracy_vp = (vp_logits.argmax(dim=1) == batch.vp.long()).float().mean()

    return BcLossResult(
        loss=loss,
        nll_ko=nll_ko.detach(),
        nll_action=nll_action.detach(),
        nll_vp=nll_vp.detach(),
        accuracy_ko=accuracy_ko.detach(),
        accuracy_vp=accuracy_vp.detach(),
    )
