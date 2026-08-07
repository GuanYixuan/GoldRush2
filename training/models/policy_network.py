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
MOVE_BUDGET = 6
ACTION_COUNT = 5
KO_COUNT = 14
OBSTACLE_KNOWN_CHANNEL = 20
OBSTACLE_CHANNEL = 21
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
    decoder_hidden: int = 128
    decoder_embedding: int = 16
    activation: str = "silu"


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
    ko_entropy: Tensor
    vp_entropy: Tensor


@dataclass(frozen=True)
class PolicyEvaluation:
    logprob: Tensor
    value: Tensor
    normalized_entropy: Tensor
    action_entropy: Tensor
    ko_entropy: Tensor
    vp_entropy: Tensor


@dataclass(frozen=True)
class PolicyBcEvaluation:
    ko_logprob: Tensor
    action_logprob: Tensor
    vp_logprob: Tensor
    value: Tensor
    normalized_entropy: Tensor
    action_entropy: Tensor
    ko_entropy: Tensor
    vp_entropy: Tensor

    @property
    def logprob(self) -> Tensor:
        return self.ko_logprob + self.action_logprob + self.vp_logprob


@dataclass(frozen=True)
class _EncodedState:
    spatial_features: Tensor
    actor_context: Tensor
    known_obstacles: Tensor
    unit0_position: Tensor
    unit1_position: Tensor


@dataclass(frozen=True)
class _EncoderOutput:
    spatial_features: Tensor
    avg: Tensor
    max_pool: Tensor
    unit0: Tensor
    unit1: Tensor
    known_obstacles: Tensor
    unit0_position: Tensor
    unit1_position: Tensor


@dataclass(frozen=True)
class _DecodedActions:
    actions: Tensor
    logprob: Tensor
    entropy: Tensor
    all_forced_actions_valid: Tensor
    final_unit0_position: Tensor
    final_unit1_position: Tensor


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


class AutoregressiveGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, hidden_size * 3)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size * 3)

    def forward(self, inputs: Tensor, hidden: Tensor) -> Tensor:
        input_gates = self.input_linear(inputs)
        hidden_gates = self.hidden_linear(hidden)
        width = self.hidden_size
        reset = torch.sigmoid(input_gates[:, :width] + hidden_gates[:, :width])
        update = torch.sigmoid(input_gates[:, width : 2 * width] + hidden_gates[:, width : 2 * width])
        candidate = torch.tanh(input_gates[:, 2 * width :] + reset * hidden_gates[:, 2 * width :])
        return candidate + update * (hidden - candidate)


class GoldRushFeatureEncoder(nn.Module):
    def __init__(self, config: PolicyNetworkConfig) -> None:
        super().__init__()
        width = config.width
        activation = config.activation
        self.stem = nn.Sequential(
            nn.Conv2d(config.spatial_channels, width, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            _activation(activation),
        )

        scalar_layers: list[nn.Module] = []
        input_dim = config.scalar_features
        for hidden_dim in config.scalar_hidden:
            scalar_layers.extend((nn.Linear(input_dim, hidden_dim), _activation(activation)))
            input_dim = hidden_dim
        scalar_layers.append(nn.Linear(input_dim, width * 2))
        self.scalar_tower = nn.Sequential(*scalar_layers)

        self.blocks = nn.ModuleList(
            SEResidualBlock(width, config.se_reduction, activation)
            for _ in range(config.residual_blocks)
        )

    def forward(self, spatial_planes: Tensor, scalars: Tensor) -> _EncoderOutput:
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
        known_obstacles = (
            (spatial_planes[:, OBSTACLE_KNOWN_CHANNEL] > 0.5)
            & (spatial_planes[:, OBSTACLE_CHANNEL] > 0.5)
        ).flatten(1)
        return _EncoderOutput(
            spatial_features=h,
            avg=avg,
            max_pool=max_pool,
            unit0=unit0,
            unit1=unit1,
            known_obstacles=known_obstacles,
            unit0_position=spatial_planes[:, OWN_UNIT0_CHANNEL].flatten(1).argmax(dim=1),
            unit1_position=spatial_planes[:, OWN_UNIT1_CHANNEL].flatten(1).argmax(dim=1),
        )


class GoldRushPolicyNetwork(nn.Module):
    def __init__(self, config: PolicyNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyNetworkConfig()
        _validate_config(self.config)

        width = self.config.width
        activation = self.config.activation
        embedding = self.config.decoder_embedding
        self.actor_encoder = GoldRushFeatureEncoder(self.config)
        self.critic_encoder = GoldRushFeatureEncoder(self.config)

        self.actor_mlp = nn.Sequential(
            nn.Linear(width * 4, self.config.actor_hidden),
            _activation(activation),
            nn.Linear(self.config.actor_hidden, self.config.actor_hidden),
            _activation(activation),
        )
        self.ko_head = nn.Linear(self.config.actor_hidden, KO_COUNT)
        self.vp_head = nn.Linear(self.config.actor_hidden, 3)

        self.ko_embedding = nn.Embedding(KO_COUNT, embedding)
        self.role_embedding = nn.Embedding(2, embedding)
        self.previous_action_embedding = nn.Embedding(ACTION_COUNT + 1, embedding)
        self.step_embedding = nn.Embedding(MOVE_BUDGET, embedding)
        decoder_input = self.config.actor_hidden + width * 2 + embedding * 4
        self.decoder_initial = nn.Linear(self.config.actor_hidden, self.config.decoder_hidden)
        self.decoder = AutoregressiveGRUCell(decoder_input, self.config.decoder_hidden)
        self.decoder_action_head = nn.Linear(self.config.decoder_hidden, ACTION_COUNT)

        self.critic_mlp = nn.Sequential(
            nn.Linear(width * 2, self.config.critic_hidden[0]),
            _activation(activation),
            nn.Linear(self.config.critic_hidden[0], self.config.critic_hidden[1]),
            _activation(activation),
            nn.Linear(self.config.critic_hidden[1], 1),
        )

        execution_roles, official_slots = _execution_tables()
        self.register_buffer("execution_role_table", execution_roles, persistent=True)
        self.register_buffer("official_slot_table", official_slots, persistent=True)
        self.register_buffer("action_row_delta", torch.tensor((-1, 1, 0, 0, 0)), persistent=True)
        self.register_buffer("action_col_delta", torch.tensor((0, 0, -1, 1, 0)), persistent=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        self._reset_encoder_specials(self.actor_encoder)
        self._reset_encoder_specials(self.critic_encoder)

        nn.init.xavier_uniform_(self.decoder.input_linear.weight)
        nn.init.orthogonal_(self.decoder.hidden_linear.weight)
        nn.init.zeros_(self.decoder.input_linear.bias)
        nn.init.zeros_(self.decoder.hidden_linear.bias)
        for head in (self.ko_head, self.vp_head, self.decoder_action_head):
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

    @staticmethod
    def _reset_encoder_specials(encoder: GoldRushFeatureEncoder) -> None:
        film_last = encoder.scalar_tower[-1]
        if not isinstance(film_last, nn.Linear):
            raise TypeError("scalar_tower final module must be nn.Linear")
        nn.init.zeros_(film_last.weight)
        nn.init.zeros_(film_last.bias)

        for block in encoder.blocks:
            nn.init.orthogonal_(block.conv2.weight, gain=0.1)
            nn.init.zeros_(block.conv2.bias)
            nn.init.zeros_(block.se_fc2.weight)
            nn.init.zeros_(block.se_fc2.bias)

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        modules = (
            self.actor_encoder,
            self.actor_mlp,
            self.ko_head,
            self.vp_head,
            self.ko_embedding,
            self.role_embedding,
            self.previous_action_embedding,
            self.step_embedding,
            self.decoder_initial,
            self.decoder,
            self.decoder_action_head,
        )
        return tuple(parameter for module in modules for parameter in module.parameters())

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.critic_encoder.parameters()) + tuple(self.critic_mlp.parameters())

    def copy_actor_encoder_to_critic(self) -> None:
        self.critic_encoder.load_state_dict(self.actor_encoder.state_dict())

    def forward(self, spatial_planes: Tensor, scalars: Tensor) -> PolicyAction:
        return self.act(spatial_planes, scalars, deterministic=True)

    def act(self, spatial_planes: Tensor, scalars: Tensor, *, deterministic: bool = False) -> PolicyAction:
        self._validate_inputs(spatial_planes, scalars)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        encoded = self._encode_actor(spatial_planes, scalars)
        value = self._critic_value(spatial_planes, scalars)
        ko_logits = self.ko_head(encoded.actor_context)
        vp_logits = self.vp_head(encoded.actor_context)
        ko, ko_logprob = _select_logits(ko_logits, deterministic=deterministic)
        vp, vp_logprob = _select_logits(vp_logits, deterministic=deterministic)
        decoded = self._decode(encoded, ko, deterministic=deterministic, forced_actions=None)

        ko_entropy = _normalized_entropy(ko_logits, math.log(KO_COUNT))
        vp_entropy = _normalized_entropy(vp_logits, math.log(3.0))
        return PolicyAction(
            actions=decoded.actions,
            k=torch.div(ko, 2, rounding_mode="floor"),
            order=torch.remainder(ko, 2),
            vp=vp,
            logprob=ko_logprob + vp_logprob + decoded.logprob,
            value=value,
            normalized_entropy=0.0100 * decoded.entropy + 0.0040 * ko_entropy + 0.0003 * vp_entropy,
            action_entropy=decoded.entropy,
            ko_entropy=ko_entropy,
            vp_entropy=vp_entropy,
        )

    def evaluate_actions(
        self,
        spatial_planes: Tensor,
        scalars: Tensor,
        actions: Tensor,
        k: Tensor,
        order: Tensor,
        vp: Tensor,
    ) -> PolicyEvaluation:
        evaluation = self.evaluate_bc_actions(spatial_planes, scalars, actions, k, order, vp)
        return PolicyEvaluation(
            logprob=evaluation.logprob,
            value=evaluation.value,
            normalized_entropy=evaluation.normalized_entropy,
            action_entropy=evaluation.action_entropy,
            ko_entropy=evaluation.ko_entropy,
            vp_entropy=evaluation.vp_entropy,
        )

    def evaluate_bc_actions(
        self,
        spatial_planes: Tensor,
        scalars: Tensor,
        actions: Tensor,
        k: Tensor,
        order: Tensor,
        vp: Tensor,
    ) -> PolicyBcEvaluation:
        self._validate_action_inputs(spatial_planes, actions, k, order, vp)
        self._validate_inputs(spatial_planes, scalars)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        encoded = self._encode_actor(spatial_planes, scalars)
        value = self._critic_value(spatial_planes, scalars)
        ko = 2 * k.long() + order.long()
        ko_logits = self.ko_head(encoded.actor_context)
        vp_logits = self.vp_head(encoded.actor_context)
        decoded = self._decode(encoded, ko, deterministic=False, forced_actions=actions.long())
        if not bool(decoded.all_forced_actions_valid.all().item()):
            raise ValueError("teacher-forced action is invalid under the autoregressive movement mask")

        ko_logprob = _logprob(ko_logits, ko)
        vp_logprob = _logprob(vp_logits, vp.long())
        ko_entropy = _normalized_entropy(ko_logits, math.log(KO_COUNT))
        vp_entropy = _normalized_entropy(vp_logits, math.log(3.0))
        return PolicyBcEvaluation(
            ko_logprob=ko_logprob,
            action_logprob=decoded.logprob,
            vp_logprob=vp_logprob,
            value=value,
            normalized_entropy=0.0100 * decoded.entropy + 0.0040 * ko_entropy + 0.0003 * vp_entropy,
            action_entropy=decoded.entropy,
            ko_entropy=ko_entropy,
            vp_entropy=vp_entropy,
        )

    def _encode(self, spatial_planes: Tensor, scalars: Tensor) -> _EncodedState:
        self._validate_inputs(spatial_planes, scalars)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        return self._encode_actor(spatial_planes, scalars)

    def _encode_actor(self, spatial_planes: Tensor, scalars: Tensor) -> _EncodedState:
        encoded = self.actor_encoder(spatial_planes, scalars)
        actor_context = self.actor_mlp(
            torch.cat((encoded.avg, encoded.max_pool, encoded.unit0, encoded.unit1), dim=1)
        )
        return _EncodedState(
            spatial_features=encoded.spatial_features,
            actor_context=actor_context,
            known_obstacles=encoded.known_obstacles,
            unit0_position=encoded.unit0_position,
            unit1_position=encoded.unit1_position,
        )

    def _critic_value(self, spatial_planes: Tensor, scalars: Tensor) -> Tensor:
        encoded = self.critic_encoder(spatial_planes, scalars)
        return self.critic_mlp(torch.cat((encoded.avg, encoded.max_pool), dim=1)).squeeze(-1)

    def _decode(
        self,
        encoded: _EncodedState,
        ko: Tensor,
        *,
        deterministic: bool,
        forced_actions: Tensor | None,
    ) -> _DecodedActions:
        batch_size = encoded.actor_context.shape[0]
        roles = self.execution_role_table[ko]
        official_slots = self.official_slot_table[ko]
        ko_embedding = self.ko_embedding(ko)
        hidden = torch.tanh(self.decoder_initial(encoded.actor_context))
        previous_action = torch.full(
            (batch_size,), ACTION_COUNT, dtype=torch.long, device=encoded.actor_context.device
        )
        unit0_position = encoded.unit0_position
        unit1_position = encoded.unit1_position
        execution_actions: list[Tensor] = []
        logprobs: list[Tensor] = []
        entropies: list[Tensor] = []
        all_forced_actions_valid = torch.ones(batch_size, dtype=torch.bool, device=encoded.actor_context.device)

        for step in range(MOVE_BUDGET):
            role = roles[:, step]
            current_position = torch.where(role == 0, unit0_position, unit1_position)
            other_position = torch.where(role == 0, unit1_position, unit0_position)
            current_local = _gather_position(encoded.spatial_features, current_position)
            other_local = _gather_position(encoded.spatial_features, other_position)
            step_index = torch.full((batch_size,), step, dtype=torch.long, device=role.device)
            decoder_input = torch.cat(
                (
                    encoded.actor_context,
                    ko_embedding,
                    self.role_embedding(role),
                    current_local,
                    other_local,
                    self.previous_action_embedding(previous_action),
                    self.step_embedding(step_index),
                ),
                dim=1,
            )
            hidden = self.decoder(decoder_input, hidden)
            logits = self.decoder_action_head(hidden)
            valid_actions, candidate_positions = self._movement_candidates(
                current_position,
                other_position,
                encoded.known_obstacles,
            )
            masked_logits = logits.masked_fill(~valid_actions, torch.finfo(logits.dtype).min)
            if forced_actions is None:
                selected, selected_logprob = _select_logits(masked_logits, deterministic=deterministic)
            else:
                selected = forced_actions.gather(1, official_slots[:, step].unsqueeze(1)).squeeze(1)
                selected_logprob = _logprob(masked_logits, selected)
                selected_valid = valid_actions.gather(1, selected.unsqueeze(1)).squeeze(1)
                all_forced_actions_valid = all_forced_actions_valid & selected_valid

            selected_position = candidate_positions.gather(1, selected.unsqueeze(1)).squeeze(1)
            selected_valid = valid_actions.gather(1, selected.unsqueeze(1)).squeeze(1)
            next_position = torch.where(selected_valid, selected_position, current_position)
            unit0_position = torch.where(role == 0, next_position, unit0_position)
            unit1_position = torch.where(role == 1, next_position, unit1_position)
            execution_actions.append(selected)
            logprobs.append(selected_logprob)
            entropies.append(_normalized_entropy(masked_logits, math.log(ACTION_COUNT)))
            previous_action = selected

        actions_in_execution_order = torch.stack(execution_actions, dim=1)
        official_actions = torch.zeros_like(actions_in_execution_order).scatter(
            1, official_slots, actions_in_execution_order
        )
        return _DecodedActions(
            actions=official_actions,
            logprob=torch.stack(logprobs, dim=1).sum(dim=1),
            entropy=torch.stack(entropies, dim=1).mean(dim=1),
            all_forced_actions_valid=all_forced_actions_valid,
            final_unit0_position=unit0_position,
            final_unit1_position=unit1_position,
        )

    def _movement_candidates(
        self,
        current_position: Tensor,
        other_position: Tensor,
        known_obstacles: Tensor,
    ) -> tuple[Tensor, Tensor]:
        row = torch.div(current_position, GRID_SIZE, rounding_mode="floor")
        col = torch.remainder(current_position, GRID_SIZE)
        candidate_row = row.unsqueeze(1) + self.action_row_delta.unsqueeze(0)
        candidate_col = col.unsqueeze(1) + self.action_col_delta.unsqueeze(0)
        in_bounds = (
            (candidate_row >= 0)
            & (candidate_row < GRID_SIZE)
            & (candidate_col >= 0)
            & (candidate_col < GRID_SIZE)
        )
        safe_row = candidate_row.clamp(0, GRID_SIZE - 1)
        safe_col = candidate_col.clamp(0, GRID_SIZE - 1)
        candidate_position = safe_row * GRID_SIZE + safe_col
        obstacle = known_obstacles.gather(1, candidate_position)
        blocked_by_unit = candidate_position == other_position.unsqueeze(1)
        valid = in_bounds & ~obstacle & ~blocked_by_unit
        return valid, candidate_position

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

    @staticmethod
    def _validate_action_inputs(
        spatial_planes: Tensor,
        actions: Tensor,
        k: Tensor,
        order: Tensor,
        vp: Tensor,
    ) -> None:
        batch_size = spatial_planes.shape[0]
        expected = {
            "actions": (batch_size, MOVE_BUDGET),
            "k": (batch_size,),
            "order": (batch_size,),
            "vp": (batch_size,),
        }
        for name, shape in expected.items():
            value = {"actions": actions, "k": k, "order": order, "vp": vp}[name]
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")


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
                action = self.model.act(spatial_planes, scalars, deterministic=self.deterministic)
                if not policy_action_is_finite(action):
                    raise ValueError("policy network produced NaN or Inf")
            self.last_action = action
            self.last_fallback_error = None
            return policy_action_to_game_output(action)
        except Exception as exc:  # pragma: no cover - tested through explicit NaN output helper
            self.last_action = None
            self.last_fallback_error = str(exc)
            return safe_game_output()


def policy_action_to_game_output(action: PolicyAction, *, batch_index: int = 0) -> GameOutput:
    if action.actions.ndim != 2 or action.actions.shape[1] != MOVE_BUDGET:
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
    return GameOutput(actions=(int(Action.STAY),) * MOVE_BUDGET, k=3, order=0, vp=0)


def policy_action_is_finite(action: PolicyAction) -> bool:
    checks = torch.stack(
        [
            torch.isfinite(tensor).all()
            for tensor in (
                action.logprob,
                action.value,
                action.normalized_entropy,
                action.action_entropy,
                action.ko_entropy,
                action.vp_entropy,
            )
        ]
    )
    return bool(checks.all().item())


def _execution_tables() -> tuple[Tensor, Tensor]:
    execution_roles: list[list[int]] = []
    official_slots: list[list[int]] = []
    for k in range(MOVE_BUDGET + 1):
        for order in range(2):
            if order == 0:
                execution_roles.append([0] * k + [1] * (MOVE_BUDGET - k))
                official_slots.append(list(range(k)) + list(range(k, MOVE_BUDGET)))
            else:
                execution_roles.append([1] * (MOVE_BUDGET - k) + [0] * k)
                official_slots.append(list(range(k, MOVE_BUDGET)) + list(range(k)))
    return torch.tensor(execution_roles, dtype=torch.long), torch.tensor(official_slots, dtype=torch.long)


def _select_logits(logits: Tensor, *, deterministic: bool) -> tuple[Tensor, Tensor]:
    if deterministic:
        selected = logits.argmax(dim=-1)
    else:
        probabilities = torch.softmax(logits, dim=-1)
        selected = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]), num_samples=1
        ).reshape(logits.shape[:-1])
    return selected, _logprob(logits, selected)


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


def _gather_position(h: Tensor, position: Tensor) -> Tensor:
    flat = h.flatten(2)
    index = position[:, None, None].expand(-1, h.shape[1], 1)
    return flat.gather(2, index).squeeze(2)


def _validate_config(config: PolicyNetworkConfig) -> None:
    if config.spatial_channels != SPATIAL_CHANNELS:
        raise ValueError(f"spatial_channels must be {SPATIAL_CHANNELS}, got {config.spatial_channels}")
    if config.scalar_features != SCALAR_FEATURES:
        raise ValueError(f"scalar_features must be {SCALAR_FEATURES}, got {config.scalar_features}")
    if config.width <= 0:
        raise ValueError(f"width must be positive, got {config.width}")
    if config.residual_blocks <= 0:
        raise ValueError(f"residual_blocks must be positive, got {config.residual_blocks}")
    if config.decoder_hidden <= 0 or config.decoder_embedding <= 0:
        raise ValueError("decoder_hidden and decoder_embedding must be positive")
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
