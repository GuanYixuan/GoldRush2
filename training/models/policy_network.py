from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from simulator.types import Action, GameOutput


FEATURE_SCHEMA = "goldrush2_feature_v1"
SPATIAL_CHANNELS = 38
SCALAR_FEATURES = 10
GRID_SIZE = 17
OWN_UNIT0_CHANNEL = 24
OWN_UNIT1_CHANNEL = 25


@dataclass(frozen=True)
class PolicyNetworkConfig:
    spatial_channels: int = SPATIAL_CHANNELS
    scalar_features: int = SCALAR_FEATURES
    width: int = 96
    residual_blocks: int = 8
    se_reduction: int = 4
    scalar_hidden: tuple[int, int] = (96, 96)
    actor_hidden: int = 256
    critic_hidden: tuple[int, int] = (256, 128)
    activation: str = "silu"


@dataclass(frozen=True)
class PolicyNetworkOutput:
    action_logits: Tensor
    k_logits: Tensor
    order_logits: Tensor
    vp_logits: Tensor
    value: Tensor


@dataclass(frozen=True)
class PolicyAction:
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    logprob: Tensor
    value: Tensor
    normalized_entropy: Tensor
    action_entropy: Tensor
    k_entropy: Tensor
    order_entropy: Tensor
    vp_entropy: Tensor


class SEResidualBlock(nn.Module):
    def __init__(self, width: int, reduction: int, activation: str) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if reduction <= 0 or width % reduction != 0:
            raise ValueError(f"reduction must divide width, got width={width}, reduction={reduction}")
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.se_fc1 = nn.Linear(width, width // reduction)
        self.se_fc2 = nn.Linear(width // reduction, width)
        self.activation = _activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.conv1(x)
        residual = self.activation(residual)
        residual = self.conv2(residual)

        gate = residual.mean(dim=(-2, -1))
        gate = self.activation(self.se_fc1(gate))
        gate = torch.sigmoid(self.se_fc2(gate)).unsqueeze(-1).unsqueeze(-1)
        return self.activation(x + residual * gate)


class GoldRushPolicyNetwork(nn.Module):
    def __init__(self, config: PolicyNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyNetworkConfig()
        _validate_config(self.config)

        width = self.config.width
        activation = self.config.activation
        self.stem = nn.Sequential(
            nn.Conv2d(self.config.spatial_channels, width, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            _activation(activation),
        )

        scalar_layers: list[nn.Module] = []
        input_dim = self.config.scalar_features
        for hidden_dim in self.config.scalar_hidden:
            scalar_layers.extend((nn.Linear(input_dim, hidden_dim), _activation(activation)))
            input_dim = hidden_dim
        scalar_layers.append(nn.Linear(input_dim, width * 2))
        self.scalar_tower = nn.Sequential(*scalar_layers)

        self.blocks = nn.ModuleList(
            SEResidualBlock(width, self.config.se_reduction, activation)
            for _ in range(self.config.residual_blocks)
        )

        actor_input_dim = width * 4
        self.actor_mlp = nn.Sequential(
            nn.Linear(actor_input_dim, self.config.actor_hidden),
            _activation(activation),
            nn.Linear(self.config.actor_hidden, self.config.actor_hidden),
            _activation(activation),
        )
        self.action_head = nn.Linear(self.config.actor_hidden, 6 * 5)
        self.k_head = nn.Linear(self.config.actor_hidden, 7)
        self.order_head = nn.Linear(self.config.actor_hidden, 2)
        self.vp_head = nn.Linear(self.config.actor_hidden, 3)

        self.critic_mlp = nn.Sequential(
            nn.Linear(width * 2, self.config.critic_hidden[0]),
            _activation(activation),
            nn.Linear(self.config.critic_hidden[0], self.config.critic_hidden[1]),
            _activation(activation),
            nn.Linear(self.config.critic_hidden[1], 1),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

        film_last = self.scalar_tower[-1]
        if not isinstance(film_last, nn.Linear):
            raise TypeError("scalar_tower final module must be nn.Linear")
        nn.init.zeros_(film_last.weight)
        nn.init.zeros_(film_last.bias)

        for block in self.blocks:
            nn.init.orthogonal_(block.conv2.weight, gain=0.1)
            nn.init.zeros_(block.conv2.bias)
            nn.init.zeros_(block.se_fc2.weight)
            nn.init.zeros_(block.se_fc2.bias)

        for head in (self.action_head, self.k_head, self.order_head, self.vp_head):
            nn.init.normal_(head.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.bias)

        vp_prior = torch.log(torch.tensor([0.90, 0.07, 0.03], dtype=self.vp_head.bias.dtype))
        with torch.no_grad():
            self.vp_head.bias.copy_(vp_prior)

        value_head = self.critic_mlp[-1]
        if not isinstance(value_head, nn.Linear):
            raise TypeError("critic_mlp final module must be nn.Linear")
        nn.init.normal_(value_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(value_head.bias)

    def forward(self, spatial_planes: Tensor, scalars: Tensor) -> PolicyNetworkOutput:
        self._validate_inputs(spatial_planes, scalars)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()

        h = self.stem(spatial_planes)
        film = self.scalar_tower(scalars)
        gamma, beta = film.chunk(2, dim=1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for block in self.blocks:
            h = block(h)

        avg = h.mean(dim=(-2, -1))
        max_pool = h.amax(dim=(-2, -1))
        unit0 = _gather_unit(h, spatial_planes[:, OWN_UNIT0_CHANNEL])
        unit1 = _gather_unit(h, spatial_planes[:, OWN_UNIT1_CHANNEL])
        actor_hidden = self.actor_mlp(torch.cat((avg, max_pool, unit0, unit1), dim=1))
        critic_input = torch.cat((avg, max_pool), dim=1)

        return PolicyNetworkOutput(
            action_logits=self.action_head(actor_hidden).view(-1, 6, 5),
            k_logits=self.k_head(actor_hidden),
            order_logits=self.order_head(actor_hidden),
            vp_logits=self.vp_head(actor_hidden),
            value=self.critic_mlp(critic_input).squeeze(-1),
        )

    def _validate_inputs(self, spatial_planes: Tensor, scalars: Tensor) -> None:
        if spatial_planes.ndim != 4:
            raise ValueError(f"spatial_planes must have shape BxCx17x17, got {tuple(spatial_planes.shape)}")
        if scalars.ndim != 2:
            raise ValueError(f"scalars must have shape Bx10, got {tuple(scalars.shape)}")
        if spatial_planes.shape[1:] != (self.config.spatial_channels, GRID_SIZE, GRID_SIZE):
            raise ValueError(
                "spatial_planes must have shape "
                f"Bx{self.config.spatial_channels}x{GRID_SIZE}x{GRID_SIZE}, got {tuple(spatial_planes.shape)}"
            )
        if scalars.shape[1] != self.config.scalar_features:
            raise ValueError(f"scalars must have {self.config.scalar_features} features, got {tuple(scalars.shape)}")
        if spatial_planes.shape[0] != scalars.shape[0]:
            raise ValueError(
                f"batch size mismatch: spatial_planes has {spatial_planes.shape[0]}, scalars has {scalars.shape[0]}"
            )


class TorchFeaturePolicy:
    """Eval/smoke adapter from policy_runtime feature dict to GameOutput."""

    def __init__(
        self,
        model: GoldRushPolicyNetwork,
        *,
        deterministic: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.deterministic = deterministic
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.last_action: PolicyAction | None = None
        self.last_fallback_error: str | None = None

    def __call__(self, features: dict[str, object]) -> GameOutput:
        try:
            if features.get("feature_schema") != FEATURE_SCHEMA:
                raise ValueError(f"unexpected feature schema: {features.get('feature_schema')!r}")
            spatial_planes = torch.as_tensor(features["planes"], dtype=torch.float32, device=self.device).unsqueeze(0)
            scalars = torch.as_tensor(features["scalars"], dtype=torch.float32, device=self.device).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                output = self.model(spatial_planes, scalars)
                if not policy_output_is_finite(output):
                    raise ValueError("policy network produced NaN or Inf")
                action = argmax_action(output) if self.deterministic else sample_action(output)
            self.last_action = action
            self.last_fallback_error = None
            return policy_action_to_game_output(action)
        except Exception as exc:  # pragma: no cover - tested through explicit NaN output helper
            self.last_action = None
            self.last_fallback_error = str(exc)
            return safe_game_output()


def sample_action(output: PolicyNetworkOutput) -> PolicyAction:
    _validate_output(output)
    actions, action_logprob = _sample_logits(output.action_logits)
    k, k_logprob = _sample_logits(output.k_logits)
    order, order_logprob = _sample_logits(output.order_logits)
    vp, vp_logprob = _sample_logits(output.vp_logits)
    return _make_policy_action(output, actions, k, order, vp, action_logprob, k_logprob, order_logprob, vp_logprob)


def argmax_action(output: PolicyNetworkOutput) -> PolicyAction:
    _validate_output(output)
    actions = output.action_logits.argmax(dim=-1)
    k = output.k_logits.argmax(dim=-1)
    order = output.order_logits.argmax(dim=-1)
    vp = output.vp_logits.argmax(dim=-1)
    action_logprob = _logprob(output.action_logits, actions)
    k_logprob = _logprob(output.k_logits, k)
    order_logprob = _logprob(output.order_logits, order)
    vp_logprob = _logprob(output.vp_logits, vp)
    return _make_policy_action(output, actions, k, order, vp, action_logprob, k_logprob, order_logprob, vp_logprob)


def policy_action_to_game_output(action: PolicyAction, *, batch_index: int = 0) -> GameOutput:
    if action.actions.ndim != 2 or action.actions.shape[1] != 6:
        raise ValueError(f"actions must have shape Bx6, got {tuple(action.actions.shape)}")
    batch_size = action.actions.shape[0]
    if not 0 <= batch_index < batch_size:
        raise IndexError(f"batch_index {batch_index} out of range for batch size {batch_size}")
    actions = tuple(int(value) for value in action.actions[batch_index].detach().cpu().tolist())
    return GameOutput(
        actions=actions,
        k=int(action.k[batch_index].detach().cpu().item()),
        order=int(action.order[batch_index].detach().cpu().item()),
        vp=int(action.vp[batch_index].detach().cpu().item()),
    )


def safe_game_output() -> GameOutput:
    return GameOutput(actions=(int(Action.STAY),) * 6, k=3, order=0, vp=0)


def policy_output_is_finite(output: PolicyNetworkOutput) -> bool:
    return all(
        torch.isfinite(tensor).all().item()
        for tensor in (output.action_logits, output.k_logits, output.order_logits, output.vp_logits, output.value)
    )


def _make_policy_action(
    output: PolicyNetworkOutput,
    actions: Tensor,
    k: Tensor,
    order: Tensor,
    vp: Tensor,
    action_logprob: Tensor,
    k_logprob: Tensor,
    order_logprob: Tensor,
    vp_logprob: Tensor,
) -> PolicyAction:
    action_entropy = _normalized_entropy(output.action_logits, math.log(5.0)).mean(dim=1)
    k_entropy = _normalized_entropy(output.k_logits, math.log(7.0))
    order_entropy = _normalized_entropy(output.order_logits, math.log(2.0))
    vp_entropy = _normalized_entropy(output.vp_logits, math.log(3.0))
    normalized_entropy = (
        0.0100 * action_entropy
        + 0.0030 * k_entropy
        + 0.0010 * order_entropy
        + 0.0003 * vp_entropy
    )
    return PolicyAction(
        actions=actions,
        k=k,
        order=order,
        vp=vp,
        logprob=action_logprob.sum(dim=1) + k_logprob + order_logprob + vp_logprob,
        value=output.value,
        normalized_entropy=normalized_entropy,
        action_entropy=action_entropy,
        k_entropy=k_entropy,
        order_entropy=order_entropy,
        vp_entropy=vp_entropy,
    )


def _sample_logits(logits: Tensor) -> tuple[Tensor, Tensor]:
    probabilities = torch.softmax(logits, dim=-1)
    sample = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        num_samples=1,
    ).reshape(logits.shape[:-1])
    return sample, _logprob(logits, sample)


def _logprob(logits: Tensor, values: Tensor) -> Tensor:
    log_probabilities = F.log_softmax(logits, dim=-1)
    return log_probabilities.gather(dim=-1, index=values.unsqueeze(-1)).squeeze(-1)


def _normalized_entropy(logits: Tensor, denominator: float) -> Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = F.log_softmax(logits, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1) / denominator


def _gather_unit(h: Tensor, unit_mask: Tensor) -> Tensor:
    mask = unit_mask.unsqueeze(1)
    numerator = (h * mask).sum(dim=(-2, -1))
    denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return numerator / denominator


def _validate_output(output: PolicyNetworkOutput) -> None:
    batch_size = output.value.shape[0]
    expected = {
        "action_logits": (batch_size, 6, 5),
        "k_logits": (batch_size, 7),
        "order_logits": (batch_size, 2),
        "vp_logits": (batch_size, 3),
        "value": (batch_size,),
    }
    for name, shape in expected.items():
        tensor = getattr(output, name)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")


def _validate_config(config: PolicyNetworkConfig) -> None:
    if config.spatial_channels != SPATIAL_CHANNELS:
        raise ValueError(f"spatial_channels must be {SPATIAL_CHANNELS}, got {config.spatial_channels}")
    if config.scalar_features != SCALAR_FEATURES:
        raise ValueError(f"scalar_features must be {SCALAR_FEATURES}, got {config.scalar_features}")
    if config.width <= 0:
        raise ValueError(f"width must be positive, got {config.width}")
    if config.residual_blocks <= 0:
        raise ValueError(f"residual_blocks must be positive, got {config.residual_blocks}")
    if len(config.scalar_hidden) != 2:
        raise ValueError("scalar_hidden must contain exactly two hidden sizes")
    if len(config.critic_hidden) != 2:
        raise ValueError("critic_hidden must contain exactly two hidden sizes")
    if config.activation not in {"silu", "relu"}:
        raise ValueError(f"activation must be 'silu' or 'relu', got {config.activation!r}")


def _activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unsupported activation: {name}")
