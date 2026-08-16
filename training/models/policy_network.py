from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from simulator.types import Action, GameOutput


FEATURE_SCHEMA = "goldrush2_feature_v2"
SPATIAL_CHANNELS = 43
SCALAR_FEATURES = 10
ACTION_HEAD_SCHEMA = "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1_vp_final_position_v1"
FAST_THRESHOLD_ACTION_HEAD_SCHEMA = "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1"
CANDIDATE_ACTION_HEAD_SCHEMA = "candidate_cell_residual_v1"
LEGACY_ACTION_HEAD_SCHEMA = "autoregressive_head_v1"
CRITIC_FEATURE_SCHEMA = "goldrush2_privileged_critic_feature_v2"
CRITIC_SPATIAL_CHANNELS = 39
CRITIC_SCALAR_FEATURES = 20
FAST_SCALAR_FEATURES = 2
INITIAL_FAST_SCALARS = (0.8, 0.25)
GRID_SIZE = 17
MOVE_BUDGET = 6
ACTION_COUNT = 5
KO_COUNT = 14
THRESHOLD_LOW = 4.0
THRESHOLD_HIGH = 30.0
FAST_THRESHOLD_BASE_LOW_BELIEF = 0.3
FAST_THRESHOLD_BASE_HIGH_BELIEF = 0.6
FAST_THRESHOLD_BASE_CONSERVATIVE = 11.0
FAST_THRESHOLD_BASE_AGGRESSIVE = 6.0
OBSTACLE_KNOWN_CHANNEL = 20
OBSTACLE_CHANNEL = 21
OWN_UNIT0_CHANNEL = 24
OWN_UNIT1_CHANNEL = 25
CRITIC_OWN_UNIT0_CHANNEL = 9
CRITIC_OWN_UNIT1_CHANNEL = 10
CRITIC_ENEMY_UNIT0_CHANNEL = 11
CRITIC_ENEMY_UNIT1_CHANNEL = 12


@dataclass(frozen=True)
class PolicyNetworkConfig:
    actor_spatial_channels: int = SPATIAL_CHANNELS
    actor_scalar_features: int = SCALAR_FEATURES
    critic_spatial_channels: int = CRITIC_SPATIAL_CHANNELS
    critic_scalar_features: int = CRITIC_SCALAR_FEATURES
    width: int = 96
    residual_blocks: int = 8
    se_reduction: int = 4
    scalar_hidden: tuple[int, int] = (96, 96)
    actor_hidden: int = 256
    critic_hidden: tuple[int, int] = (256, 128)
    decoder_hidden: int = 128
    decoder_embedding: int = 16
    action_head_schema: str = ACTION_HEAD_SCHEMA
    fast_scalar_features: int = FAST_SCALAR_FEATURES
    threshold_initial: float = 12.0
    threshold_hidden: tuple[int, int] = (128, 64)
    threshold_log_std_initial: float = -1.3
    threshold_log_std_min: float = -3.0
    threshold_log_std_max: float = 0.0
    threshold_entropy_coef: float = 0.0
    activation: str = "silu"

    @property
    def spatial_channels(self) -> int:
        return self.actor_spatial_channels

    @property
    def scalar_features(self) -> int:
        return self.actor_scalar_features


@dataclass(frozen=True)
class PolicyAction:
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    threshold_raw: Tensor
    threshold_int: Tensor
    threshold_mu_raw: Tensor
    threshold_base_raw: Tensor
    threshold_residual_raw: Tensor
    logprob: Tensor
    value: Tensor
    normalized_entropy: Tensor
    action_entropy: Tensor
    ko_entropy: Tensor
    vp_entropy: Tensor
    threshold_entropy: Tensor


@dataclass(frozen=True)
class PolicyEvaluation:
    logprob: Tensor
    value: Tensor
    normalized_entropy: Tensor
    action_entropy: Tensor
    ko_entropy: Tensor
    vp_entropy: Tensor
    threshold_entropy: Tensor


@dataclass(frozen=True)
class PolicyBcEvaluation:
    ko_logprob: Tensor
    action_logprob: Tensor
    vp_logprob: Tensor
    threshold_logprob: Tensor
    value: Tensor
    normalized_entropy: Tensor
    action_entropy: Tensor
    ko_entropy: Tensor
    vp_entropy: Tensor
    threshold_entropy: Tensor

    @property
    def logprob(self) -> Tensor:
        return self.ko_logprob + self.action_logprob + self.vp_logprob + self.threshold_logprob


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
class _CriticEncoderOutput:
    spatial_features: Tensor
    avg: Tensor
    max_pool: Tensor
    own_unit0: Tensor
    own_unit1: Tensor
    enemy_unit0: Tensor
    enemy_unit1: Tensor


@dataclass(frozen=True)
class _DecodedActions:
    actions: Tensor
    logprob: Tensor
    entropy: Tensor
    all_forced_actions_valid: Tensor
    final_unit0_position: Tensor
    final_unit1_position: Tensor


@dataclass(frozen=True)
class _ThresholdAction:
    raw: Tensor
    threshold_int: Tensor
    mu: Tensor
    base_raw: Tensor
    residual_raw: Tensor
    logprob: Tensor
    entropy: Tensor


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
    def __init__(self, config: PolicyNetworkConfig, *, spatial_channels: int, scalar_features: int) -> None:
        super().__init__()
        width = config.width
        activation = config.activation
        self.stem = nn.Sequential(
            nn.Conv2d(spatial_channels, width, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            _activation(activation),
        )

        scalar_layers: list[nn.Module] = []
        input_dim = scalar_features
        for hidden_dim in config.scalar_hidden:
            scalar_layers.extend((nn.Linear(input_dim, hidden_dim), _activation(activation)))
            input_dim = hidden_dim
        scalar_layers.append(nn.Linear(input_dim, width * 2))
        self.scalar_tower = nn.Sequential(*scalar_layers)

        self.blocks = nn.ModuleList(
            SEResidualBlock(width, config.se_reduction, activation)
            for _ in range(config.residual_blocks)
        )

    def _encode_features(self, spatial_planes: Tensor, scalars: Tensor) -> Tensor:
        h = self.stem(spatial_planes)
        film = self.scalar_tower(scalars)
        gamma, beta = film.chunk(2, dim=1)
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for block in self.blocks:
            h = block(h)
        return h


class GoldRushActorFeatureEncoder(GoldRushFeatureEncoder):
    def __init__(self, config: PolicyNetworkConfig) -> None:
        super().__init__(
            config,
            spatial_channels=config.actor_spatial_channels,
            scalar_features=config.actor_scalar_features,
        )

    def forward(self, spatial_planes: Tensor, scalars: Tensor) -> _EncoderOutput:
        h = self._encode_features(spatial_planes, scalars)
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


class GoldRushPrivilegedCriticEncoder(GoldRushFeatureEncoder):
    def __init__(self, config: PolicyNetworkConfig) -> None:
        super().__init__(
            config,
            spatial_channels=config.critic_spatial_channels,
            scalar_features=config.critic_scalar_features,
        )

    def forward(self, spatial_planes: Tensor, scalars: Tensor) -> _CriticEncoderOutput:
        h = self._encode_features(spatial_planes, scalars)
        return _CriticEncoderOutput(
            spatial_features=h,
            avg=h.mean(dim=(-2, -1)),
            max_pool=h.amax(dim=(-2, -1)),
            own_unit0=_gather_unit(h, spatial_planes[:, CRITIC_OWN_UNIT0_CHANNEL]),
            own_unit1=_gather_unit(h, spatial_planes[:, CRITIC_OWN_UNIT1_CHANNEL]),
            enemy_unit0=_gather_unit(h, spatial_planes[:, CRITIC_ENEMY_UNIT0_CHANNEL]),
            enemy_unit1=_gather_unit(h, spatial_planes[:, CRITIC_ENEMY_UNIT1_CHANNEL]),
        )


class GoldRushPolicyNetwork(nn.Module):
    def __init__(self, config: PolicyNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyNetworkConfig()
        _validate_config(self.config)

        width = self.config.width
        activation = self.config.activation
        embedding = self.config.decoder_embedding
        self.actor_encoder = GoldRushActorFeatureEncoder(self.config)
        self.critic_encoder = GoldRushPrivilegedCriticEncoder(self.config)

        self.actor_mlp = nn.Sequential(
            nn.Linear(width * 4, self.config.actor_hidden),
            _activation(activation),
            nn.Linear(self.config.actor_hidden, self.config.actor_hidden),
            _activation(activation),
        )
        self.ko_head = nn.Linear(self.config.actor_hidden, KO_COUNT)
        self.vp_head = nn.Linear(self.config.actor_hidden + width * 2, 3)

        self.ko_embedding = nn.Embedding(KO_COUNT, embedding)
        self.role_embedding = nn.Embedding(2, embedding)
        self.previous_action_embedding = nn.Embedding(ACTION_COUNT + 1, embedding)
        self.step_embedding = nn.Embedding(MOVE_BUDGET, embedding)
        decoder_input = self.config.actor_hidden + width * 2 + embedding * 4
        self.decoder_initial = nn.Linear(self.config.actor_hidden, self.config.decoder_hidden)
        self.decoder = AutoregressiveGRUCell(decoder_input, self.config.decoder_hidden)
        self.decoder_action_head = nn.Linear(self.config.decoder_hidden, ACTION_COUNT)
        self.action_candidate_embedding = nn.Embedding(ACTION_COUNT, embedding)
        self.candidate_action_head = nn.Sequential(
            nn.Linear(self.config.decoder_hidden + width + embedding, self.config.decoder_hidden),
            _activation(activation),
            nn.Linear(self.config.decoder_hidden, 1),
        )
        threshold_layers: list[nn.Module] = []
        threshold_input = self.config.actor_hidden + width * 2 + self.config.fast_scalar_features
        for hidden_dim in self.config.threshold_hidden:
            threshold_layers.extend((nn.Linear(threshold_input, hidden_dim), _activation(activation)))
            threshold_input = hidden_dim
        threshold_layers.append(nn.Linear(threshold_input, 1))
        self.threshold_mlp = nn.Sequential(*threshold_layers)
        self.threshold_log_std = nn.Parameter(torch.tensor(float(self.config.threshold_log_std_initial)))

        self.critic_mlp = nn.Sequential(
            nn.Linear(width * 6 + self.config.fast_scalar_features, self.config.critic_hidden[0]),
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
        self.register_buffer("action_id_table", torch.arange(ACTION_COUNT, dtype=torch.long), persistent=False)
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
        candidate_output = self.candidate_action_head[-1]
        if not isinstance(candidate_output, nn.Linear):
            raise TypeError("candidate_action_head final module must be nn.Linear")
        nn.init.zeros_(candidate_output.weight)
        nn.init.zeros_(candidate_output.bias)
        threshold_output = self.threshold_mlp[-1]
        if not isinstance(threshold_output, nn.Linear):
            raise TypeError("threshold_mlp final module must be nn.Linear")
        nn.init.zeros_(threshold_output.weight)
        nn.init.zeros_(threshold_output.bias)
        with torch.no_grad():
            self.threshold_log_std.fill_(float(self.config.threshold_log_std_initial))

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

    def ordinary_actor_parameters(self) -> tuple[nn.Parameter, ...]:
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
            self.action_candidate_embedding,
            self.candidate_action_head,
        )
        return tuple(parameter for module in modules for parameter in module.parameters())

    def threshold_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.threshold_mlp.parameters()) + (self.threshold_log_std,)

    def actor_parameters(self) -> tuple[nn.Parameter, ...]:
        return self.ordinary_actor_parameters() + self.threshold_parameters()

    def critic_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.critic_encoder.parameters()) + tuple(self.critic_mlp.parameters())

    def copy_actor_encoder_to_critic(self) -> None:
        raise RuntimeError("actor and privileged critic encoders use different feature schemas")

    def forward(self, spatial_planes: Tensor, scalars: Tensor, fast_scalars: Tensor | None = None) -> PolicyAction:
        return self.act_actor_only(spatial_planes, scalars, fast_scalars, deterministic=False)

    def act(
        self,
        spatial_planes: Tensor,
        scalars: Tensor,
        fast_scalars: Tensor,
        critic_planes: Tensor | None = None,
        critic_scalars: Tensor | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        sample_uniforms: Tensor | None = None,
    ) -> PolicyAction:
        if critic_scalars is None:
            if critic_planes is None:
                raise TypeError("critic_planes and critic_scalars are required")
            critic_scalars = critic_planes
            critic_planes = fast_scalars
            fast_scalars = self._default_fast_scalars(spatial_planes)
        assert critic_planes is not None
        self._validate_actor_inputs(spatial_planes, scalars)
        self._validate_fast_scalars(fast_scalars, spatial_planes)
        self._validate_critic_inputs(critic_planes, critic_scalars)
        self._validate_actor_critic_batch(spatial_planes, critic_planes)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        fast_scalars = fast_scalars.float()
        critic_planes = critic_planes.float()
        critic_scalars = critic_scalars.float()
        encoded = self._encode_actor(spatial_planes, scalars)
        value = self._critic_value(critic_planes, critic_scalars, fast_scalars)
        return self._act_from_encoded(
            encoded,
            fast_scalars=fast_scalars,
            value=value,
            deterministic=deterministic,
            generator=generator,
            sample_uniforms=sample_uniforms,
        )

    def act_actor_only(
        self,
        spatial_planes: Tensor,
        scalars: Tensor,
        fast_scalars: Tensor | None = None,
        *,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        sample_uniforms: Tensor | None = None,
    ) -> PolicyAction:
        self._validate_actor_inputs(spatial_planes, scalars)
        fast_scalars = self._default_fast_scalars(spatial_planes) if fast_scalars is None else fast_scalars
        self._validate_fast_scalars(fast_scalars, spatial_planes)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        fast_scalars = fast_scalars.float()
        encoded = self._encode_actor(spatial_planes, scalars)
        value = torch.zeros(spatial_planes.shape[0], dtype=spatial_planes.dtype, device=spatial_planes.device)
        return self._act_from_encoded(
            encoded,
            fast_scalars=fast_scalars,
            value=value,
            deterministic=deterministic,
            generator=generator,
            sample_uniforms=sample_uniforms,
        )

    def _act_from_encoded(
        self,
        encoded: _EncodedState,
        *,
        fast_scalars: Tensor,
        value: Tensor,
        deterministic: bool,
        generator: torch.Generator | None,
        sample_uniforms: Tensor | None,
    ) -> PolicyAction:
        if sample_uniforms is not None and tuple(sample_uniforms.shape) != (encoded.actor_context.shape[0], MOVE_BUDGET + 3):
            raise ValueError(
                f"sample_uniforms must have shape Bx{MOVE_BUDGET + 3}, got {tuple(sample_uniforms.shape)}"
            )
        ko_logits = self.ko_head(encoded.actor_context)
        ko, ko_logprob = _select_logits(
            ko_logits,
            deterministic=deterministic,
            generator=generator,
            sample_uniform=None if sample_uniforms is None else sample_uniforms[:, 0],
        )
        decoded = self._decode(
            encoded,
            ko,
            deterministic=deterministic,
            forced_actions=None,
            generator=generator,
            sample_uniforms=None if sample_uniforms is None else sample_uniforms[:, 2 : 2 + MOVE_BUDGET],
        )
        vp_logits = self._vp_logits(encoded, decoded)
        vp, vp_logprob = _select_logits(
            vp_logits,
            deterministic=deterministic,
            generator=generator,
            sample_uniform=None if sample_uniforms is None else sample_uniforms[:, 1],
        )
        threshold = self._select_threshold(
            encoded,
            decoded,
            fast_scalars,
            deterministic=deterministic,
            generator=generator,
            sample_uniform=None if sample_uniforms is None else sample_uniforms[:, -1],
        )

        ko_entropy = _normalized_entropy(ko_logits, math.log(KO_COUNT))
        vp_entropy = _normalized_entropy(vp_logits, math.log(3.0))
        return PolicyAction(
            actions=decoded.actions,
            k=torch.div(ko, 2, rounding_mode="floor"),
            order=torch.remainder(ko, 2),
            vp=vp,
            threshold_raw=threshold.raw,
            threshold_int=threshold.threshold_int,
            threshold_mu_raw=threshold.mu,
            threshold_base_raw=threshold.base_raw,
            threshold_residual_raw=threshold.residual_raw,
            logprob=ko_logprob + vp_logprob + decoded.logprob + threshold.logprob,
            value=value,
            normalized_entropy=(
                0.0100 * decoded.entropy
                + 0.0040 * ko_entropy
                + 0.0003 * vp_entropy
                + self.config.threshold_entropy_coef * threshold.entropy
            ),
            action_entropy=decoded.entropy,
            ko_entropy=ko_entropy,
            vp_entropy=vp_entropy,
            threshold_entropy=threshold.entropy,
        )

    def evaluate_actions(
        self,
        spatial_planes: Tensor,
        scalars: Tensor,
        fast_scalars: Tensor,
        critic_planes: Tensor,
        critic_scalars: Tensor,
        actions: Tensor,
        k: Tensor,
        order: Tensor,
        vp: Tensor,
        threshold_raw: Tensor,
    ) -> PolicyEvaluation:
        self._validate_action_inputs(spatial_planes, actions, k, order, vp)
        self._validate_actor_inputs(spatial_planes, scalars)
        self._validate_fast_scalars(fast_scalars, spatial_planes)
        self._validate_critic_inputs(critic_planes, critic_scalars)
        self._validate_actor_critic_batch(spatial_planes, critic_planes)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        fast_scalars = fast_scalars.float()
        critic_planes = critic_planes.float()
        critic_scalars = critic_scalars.float()
        evaluation = self._evaluate_actor_actions(
            spatial_planes,
            scalars,
            actions,
            k,
            order,
            vp,
            fast_scalars=fast_scalars,
            threshold_raw=threshold_raw,
        )
        return PolicyEvaluation(
            logprob=evaluation.logprob,
            value=self._critic_value(critic_planes, critic_scalars, fast_scalars),
            normalized_entropy=evaluation.normalized_entropy,
            action_entropy=evaluation.action_entropy,
            ko_entropy=evaluation.ko_entropy,
            vp_entropy=evaluation.vp_entropy,
            threshold_entropy=evaluation.threshold_entropy,
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
        self._validate_actor_inputs(spatial_planes, scalars)
        spatial_planes = spatial_planes.float()
        scalars = scalars.float()
        return self._evaluate_actor_actions(spatial_planes, scalars, actions, k, order, vp)

    def _evaluate_actor_actions(
        self,
        spatial_planes: Tensor,
        scalars: Tensor,
        actions: Tensor,
        k: Tensor,
        order: Tensor,
        vp: Tensor,
        fast_scalars: Tensor | None = None,
        threshold_raw: Tensor | None = None,
    ) -> PolicyBcEvaluation:
        encoded = self._encode_actor(spatial_planes, scalars)
        ko = 2 * k.long() + order.long()
        ko_logits = self.ko_head(encoded.actor_context)
        decoded = self._decode(encoded, ko, deterministic=False, forced_actions=actions.long())
        if not bool(decoded.all_forced_actions_valid.all().item()):
            raise ValueError("teacher-forced action is invalid under the autoregressive movement mask")
        vp_logits = self._vp_logits(encoded, decoded)

        ko_logprob = _logprob(ko_logits, ko)
        vp_logprob = _logprob(vp_logits, vp.long())
        ko_entropy = _normalized_entropy(ko_logits, math.log(KO_COUNT))
        vp_entropy = _normalized_entropy(vp_logits, math.log(3.0))
        threshold_logprob = torch.zeros(spatial_planes.shape[0], dtype=spatial_planes.dtype, device=spatial_planes.device)
        threshold_entropy = torch.zeros_like(threshold_logprob)
        if fast_scalars is not None:
            if threshold_raw is None:
                raise ValueError("threshold_raw is required when fast_scalars are provided")
            threshold = self._evaluate_threshold(encoded, decoded, fast_scalars.float(), threshold_raw.float())
            threshold_logprob = threshold.logprob
            threshold_entropy = threshold.entropy
        return PolicyBcEvaluation(
            ko_logprob=ko_logprob,
            action_logprob=decoded.logprob,
            vp_logprob=vp_logprob,
            threshold_logprob=threshold_logprob,
            value=torch.zeros(spatial_planes.shape[0], dtype=spatial_planes.dtype, device=spatial_planes.device),
            normalized_entropy=(
                0.0100 * decoded.entropy
                + 0.0040 * ko_entropy
                + 0.0003 * vp_entropy
                + self.config.threshold_entropy_coef * threshold_entropy
            ),
            action_entropy=decoded.entropy,
            ko_entropy=ko_entropy,
            vp_entropy=vp_entropy,
            threshold_entropy=threshold_entropy,
        )

    def _encode(self, spatial_planes: Tensor, scalars: Tensor) -> _EncodedState:
        self._validate_actor_inputs(spatial_planes, scalars)
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

    def _critic_value(self, critic_planes: Tensor, critic_scalars: Tensor, fast_scalars: Tensor) -> Tensor:
        encoded = self.critic_encoder(critic_planes, critic_scalars)
        return self.critic_mlp(
            torch.cat(
                (
                    encoded.avg,
                    encoded.max_pool,
                    encoded.own_unit0,
                    encoded.own_unit1,
                    encoded.enemy_unit0,
                    encoded.enemy_unit1,
                    fast_scalars,
                ),
                dim=1,
            )
        ).squeeze(-1)

    def _select_threshold(
        self,
        encoded: _EncodedState,
        decoded: _DecodedActions,
        fast_scalars: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None,
        sample_uniform: Tensor | None,
    ) -> _ThresholdAction:
        base_raw, residual_raw, mu = self._threshold_components(encoded, decoded, fast_scalars)
        log_std = self._threshold_log_std()
        if deterministic:
            raw = mu
        elif sample_uniform is not None:
            if sample_uniform.shape != mu.shape:
                raise ValueError(f"threshold sample_uniform must have shape {tuple(mu.shape)}, got {tuple(sample_uniform.shape)}")
            raw = mu + log_std.exp() * _standard_normal_icdf(sample_uniform.to(device=mu.device, dtype=mu.dtype))
        else:
            raw = mu + log_std.exp() * torch.randn(mu.shape, dtype=mu.dtype, device=mu.device, generator=generator)
        return self._threshold_action(base_raw, residual_raw, mu, log_std, raw)

    def _evaluate_threshold(
        self,
        encoded: _EncodedState,
        decoded: _DecodedActions,
        fast_scalars: Tensor,
        raw: Tensor,
    ) -> _ThresholdAction:
        base_raw, residual_raw, mu = self._threshold_components(encoded, decoded, fast_scalars)
        log_std = self._threshold_log_std()
        raw = raw.reshape(mu.shape).to(device=mu.device, dtype=mu.dtype)
        return self._threshold_action(base_raw, residual_raw, mu, log_std, raw)

    def _threshold_mu(self, encoded: _EncodedState, decoded: _DecodedActions, fast_scalars: Tensor) -> Tensor:
        return self._threshold_components(encoded, decoded, fast_scalars)[2]

    def _vp_logits(self, encoded: _EncodedState, decoded: _DecodedActions) -> Tensor:
        final_unit0_local = _gather_position(encoded.spatial_features, decoded.final_unit0_position)
        final_unit1_local = _gather_position(encoded.spatial_features, decoded.final_unit1_position)
        return self.vp_head(torch.cat((encoded.actor_context, final_unit0_local, final_unit1_local), dim=1))

    def _threshold_components(
        self,
        encoded: _EncodedState,
        decoded: _DecodedActions,
        fast_scalars: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        base_raw = self._threshold_base_raw(fast_scalars)
        residual_raw = self._threshold_residual_raw(encoded, decoded, fast_scalars)
        return base_raw, residual_raw, base_raw + residual_raw

    def _threshold_base_raw(self, fast_scalars: Tensor) -> Tensor:
        p = fast_scalars[:, 0].clamp(0.0, 1.0)
        ramp = ((p - FAST_THRESHOLD_BASE_LOW_BELIEF) / (FAST_THRESHOLD_BASE_HIGH_BELIEF - FAST_THRESHOLD_BASE_LOW_BELIEF)).clamp(
            0.0,
            1.0,
        )
        base_threshold = FAST_THRESHOLD_BASE_CONSERVATIVE + ramp * (
            FAST_THRESHOLD_BASE_AGGRESSIVE - FAST_THRESHOLD_BASE_CONSERVATIVE
        )
        return _threshold_to_raw_tensor(base_threshold)

    def _threshold_residual_raw(self, encoded: _EncodedState, decoded: _DecodedActions, fast_scalars: Tensor) -> Tensor:
        final_unit0_local = _gather_position(encoded.spatial_features, decoded.final_unit0_position)
        final_unit1_local = _gather_position(encoded.spatial_features, decoded.final_unit1_position)
        return self.threshold_mlp(
            torch.cat((encoded.actor_context, final_unit0_local, final_unit1_local, fast_scalars), dim=1)
        ).squeeze(-1)

    def _threshold_log_std(self) -> Tensor:
        return self.threshold_log_std.clamp(
            min=float(self.config.threshold_log_std_min),
            max=float(self.config.threshold_log_std_max),
        )

    def _threshold_action(
        self,
        base_raw: Tensor,
        residual_raw: Tensor,
        mu: Tensor,
        log_std: Tensor,
        raw: Tensor,
    ) -> _ThresholdAction:
        threshold = THRESHOLD_LOW + (THRESHOLD_HIGH - THRESHOLD_LOW) * torch.sigmoid(raw)
        threshold_int = torch.floor(threshold + 0.5).clamp(int(THRESHOLD_LOW), int(THRESHOLD_HIGH)).long()
        logprob = _normal_logprob(raw, mu, log_std)
        entropy = (0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std).expand_as(mu)
        return _ThresholdAction(
            raw=raw,
            threshold_int=threshold_int,
            mu=mu,
            base_raw=base_raw,
            residual_raw=residual_raw,
            logprob=logprob,
            entropy=entropy,
        )

    def _decode(
        self,
        encoded: _EncodedState,
        ko: Tensor,
        *,
        deterministic: bool,
        forced_actions: Tensor | None,
        generator: torch.Generator | None = None,
        sample_uniforms: Tensor | None = None,
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
        if sample_uniforms is not None and tuple(sample_uniforms.shape) != (batch_size, MOVE_BUDGET):
            raise ValueError(f"sample_uniforms must have shape Bx{MOVE_BUDGET}, got {tuple(sample_uniforms.shape)}")

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
            valid_actions, candidate_positions = self._movement_candidates(
                current_position,
                other_position,
                encoded.known_obstacles,
            )
            logits = self._action_logits(hidden, encoded.spatial_features, candidate_positions)
            masked_logits = logits.masked_fill(~valid_actions, torch.finfo(logits.dtype).min)
            if forced_actions is None:
                selected, selected_logprob = _select_logits(
                    masked_logits,
                    deterministic=deterministic,
                    generator=generator,
                    sample_uniform=None if sample_uniforms is None else sample_uniforms[:, step],
                )
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

    def _action_logits(self, hidden: Tensor, spatial_features: Tensor, candidate_positions: Tensor) -> Tensor:
        base_logits = self.decoder_action_head(hidden)
        candidate_local = _gather_positions(spatial_features, candidate_positions)
        batch_size = hidden.shape[0]
        hidden_expanded = hidden[:, None, :].expand(-1, ACTION_COUNT, -1)
        action_embedding = self.action_candidate_embedding(self.action_id_table)
        action_embedding = action_embedding[None, :, :].expand(batch_size, -1, -1)
        candidate_input = torch.cat((hidden_expanded, candidate_local, action_embedding), dim=-1)
        candidate_delta = self.candidate_action_head(candidate_input).squeeze(-1)
        return base_logits + candidate_delta

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

    def _validate_actor_inputs(self, spatial_planes: Tensor, scalars: Tensor) -> None:
        if spatial_planes.ndim != 4:
            raise ValueError(f"spatial_planes must have shape BxCx17x17, got {tuple(spatial_planes.shape)}")
        if scalars.ndim != 2:
            raise ValueError(f"scalars must have shape Bx10, got {tuple(scalars.shape)}")
        if spatial_planes.shape[1:] != (self.config.actor_spatial_channels, GRID_SIZE, GRID_SIZE):
            raise ValueError(
                "spatial_planes must have shape "
                f"Bx{self.config.actor_spatial_channels}x{GRID_SIZE}x{GRID_SIZE}, got {tuple(spatial_planes.shape)}"
            )
        if scalars.shape[1] != self.config.actor_scalar_features:
            raise ValueError(f"scalars must have {self.config.actor_scalar_features} features, got {tuple(scalars.shape)}")
        if spatial_planes.shape[0] != scalars.shape[0]:
            raise ValueError(
                f"batch size mismatch: spatial_planes has {spatial_planes.shape[0]}, scalars has {scalars.shape[0]}"
            )
        if spatial_planes.device != scalars.device:
            raise ValueError(
                f"device mismatch: spatial_planes is on {spatial_planes.device}, scalars is on {scalars.device}"
            )

    def _validate_critic_inputs(self, critic_planes: Tensor, critic_scalars: Tensor) -> None:
        if critic_planes.ndim != 4:
            raise ValueError(f"critic_planes must have shape BxCx17x17, got {tuple(critic_planes.shape)}")
        if critic_scalars.ndim != 2:
            raise ValueError(
                f"critic_scalars must have shape Bx{self.config.critic_scalar_features}, got {tuple(critic_scalars.shape)}"
            )
        if critic_planes.shape[1:] != (self.config.critic_spatial_channels, GRID_SIZE, GRID_SIZE):
            raise ValueError(
                "critic_planes must have shape "
                f"Bx{self.config.critic_spatial_channels}x{GRID_SIZE}x{GRID_SIZE}, got {tuple(critic_planes.shape)}"
            )
        if critic_scalars.shape[1] != self.config.critic_scalar_features:
            raise ValueError(
                f"critic_scalars must have {self.config.critic_scalar_features} features, got {tuple(critic_scalars.shape)}"
            )
        if critic_planes.shape[0] != critic_scalars.shape[0]:
            raise ValueError(
                f"batch size mismatch: critic_planes has {critic_planes.shape[0]}, "
                f"critic_scalars has {critic_scalars.shape[0]}"
            )
        if critic_planes.device != critic_scalars.device:
            raise ValueError(
                f"device mismatch: critic_planes is on {critic_planes.device}, "
                f"critic_scalars is on {critic_scalars.device}"
            )

    def _validate_fast_scalars(self, fast_scalars: Tensor, spatial_planes: Tensor) -> None:
        if fast_scalars.ndim != 2:
            raise ValueError(f"fast_scalars must have shape Bx{self.config.fast_scalar_features}, got {tuple(fast_scalars.shape)}")
        if fast_scalars.shape != (spatial_planes.shape[0], self.config.fast_scalar_features):
            raise ValueError(
                f"fast_scalars must have shape Bx{self.config.fast_scalar_features}, got {tuple(fast_scalars.shape)}"
            )
        if fast_scalars.device != spatial_planes.device:
            raise ValueError(
                f"device mismatch: fast_scalars is on {fast_scalars.device}, spatial_planes is on {spatial_planes.device}"
            )

    def _default_fast_scalars(self, spatial_planes: Tensor) -> Tensor:
        return torch.tensor(
            INITIAL_FAST_SCALARS,
            dtype=spatial_planes.dtype,
            device=spatial_planes.device,
        ).expand(spatial_planes.shape[0], -1)

    @staticmethod
    def _validate_actor_critic_batch(spatial_planes: Tensor, critic_planes: Tensor) -> None:
        if spatial_planes.shape[0] != critic_planes.shape[0]:
            raise ValueError(
                f"actor/critic batch size mismatch: actor has {spatial_planes.shape[0]}, "
                f"critic has {critic_planes.shape[0]}"
            )
        if spatial_planes.device != critic_planes.device:
            raise ValueError(
                f"actor/critic device mismatch: actor is on {spatial_planes.device}, "
                f"critic is on {critic_planes.device}"
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
                action = self.model.act_actor_only(spatial_planes, scalars, deterministic=self.deterministic)
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
                action.threshold_raw,
                action.threshold_int.float(),
                action.threshold_mu_raw,
                action.threshold_base_raw,
                action.threshold_residual_raw,
                action.threshold_entropy,
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


def _select_logits(
    logits: Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
    sample_uniform: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if deterministic:
        selected = logits.argmax(dim=-1)
    elif sample_uniform is not None:
        if sample_uniform.shape != logits.shape[:-1]:
            raise ValueError(f"sample_uniform must have shape {tuple(logits.shape[:-1])}, got {tuple(sample_uniform.shape)}")
        probabilities = torch.softmax(logits, dim=-1)
        cdf = probabilities.cumsum(dim=-1)
        sample = sample_uniform.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0 - torch.finfo(logits.dtype).eps)
        selected = torch.searchsorted(cdf.contiguous(), sample.unsqueeze(-1), right=False).squeeze(-1)
        selected = selected.clamp(max=logits.shape[-1] - 1)
    else:
        probabilities = torch.softmax(logits, dim=-1)
        selected = torch.multinomial(
            probabilities.reshape(-1, probabilities.shape[-1]),
            num_samples=1,
            generator=generator,
        ).reshape(logits.shape[:-1])
    return selected, _logprob(logits, selected)


def _logprob(logits: Tensor, values: Tensor) -> Tensor:
    log_probabilities = F.log_softmax(logits, dim=-1)
    return log_probabilities.gather(dim=-1, index=values.unsqueeze(-1)).squeeze(-1)


def _normalized_entropy(logits: Tensor, denominator: float) -> Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = F.log_softmax(logits, dim=-1)
    return -(probabilities * log_probabilities).sum(dim=-1) / denominator


def _normal_logprob(value: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    variance = torch.exp(2.0 * log_std)
    return -0.5 * ((value - mean).pow(2) / variance + 2.0 * log_std + math.log(2.0 * math.pi))


def _standard_normal_icdf(uniform: Tensor) -> Tensor:
    eps = torch.finfo(uniform.dtype).eps
    u = uniform.clamp(eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def _threshold_to_raw(threshold: float) -> float:
    if not THRESHOLD_LOW < threshold < THRESHOLD_HIGH:
        raise ValueError(f"threshold_initial must be in ({THRESHOLD_LOW}, {THRESHOLD_HIGH}), got {threshold}")
    ratio = (threshold - THRESHOLD_LOW) / (THRESHOLD_HIGH - THRESHOLD_LOW)
    return math.log(ratio / (1.0 - ratio))


def _threshold_to_raw_tensor(threshold: Tensor) -> Tensor:
    ratio = ((threshold - THRESHOLD_LOW) / (THRESHOLD_HIGH - THRESHOLD_LOW)).clamp(
        torch.finfo(threshold.dtype).eps,
        1.0 - torch.finfo(threshold.dtype).eps,
    )
    return torch.log(ratio / (1.0 - ratio))


def _gather_unit(h: Tensor, unit_mask: Tensor) -> Tensor:
    mask = unit_mask.unsqueeze(1)
    numerator = (h * mask).sum(dim=(-2, -1))
    denominator = mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return numerator / denominator


def _gather_position(h: Tensor, position: Tensor) -> Tensor:
    flat = h.flatten(2)
    index = position[:, None, None].expand(-1, h.shape[1], 1)
    return flat.gather(2, index).squeeze(2)


def _gather_positions(h: Tensor, positions: Tensor) -> Tensor:
    flat = h.flatten(2)
    index = positions[:, None, :].expand(-1, h.shape[1], -1)
    return flat.gather(2, index).transpose(1, 2)


def _validate_config(config: PolicyNetworkConfig) -> None:
    if config.actor_spatial_channels != SPATIAL_CHANNELS:
        raise ValueError(f"actor_spatial_channels must be {SPATIAL_CHANNELS}, got {config.actor_spatial_channels}")
    if config.actor_scalar_features != SCALAR_FEATURES:
        raise ValueError(f"actor_scalar_features must be {SCALAR_FEATURES}, got {config.actor_scalar_features}")
    if config.critic_spatial_channels != CRITIC_SPATIAL_CHANNELS:
        raise ValueError(
            f"critic_spatial_channels must be {CRITIC_SPATIAL_CHANNELS}, got {config.critic_spatial_channels}"
        )
    if config.critic_scalar_features != CRITIC_SCALAR_FEATURES:
        raise ValueError(
            f"critic_scalar_features must be {CRITIC_SCALAR_FEATURES}, got {config.critic_scalar_features}"
        )
    if config.action_head_schema not in {ACTION_HEAD_SCHEMA, FAST_THRESHOLD_ACTION_HEAD_SCHEMA, CANDIDATE_ACTION_HEAD_SCHEMA}:
        raise ValueError(
            "action_head_schema must be one of "
            f"{ACTION_HEAD_SCHEMA!r}, {FAST_THRESHOLD_ACTION_HEAD_SCHEMA!r}, or {CANDIDATE_ACTION_HEAD_SCHEMA!r}; "
            f"got {config.action_head_schema!r}"
        )
    if config.fast_scalar_features != FAST_SCALAR_FEATURES:
        raise ValueError(f"fast_scalar_features must be {FAST_SCALAR_FEATURES}, got {config.fast_scalar_features}")
    if not THRESHOLD_LOW < float(config.threshold_initial) < THRESHOLD_HIGH:
        raise ValueError(
            f"threshold_initial must be in ({THRESHOLD_LOW}, {THRESHOLD_HIGH}), got {config.threshold_initial}"
        )
    if len(config.threshold_hidden) != 2:
        raise ValueError("threshold_hidden must contain exactly two hidden sizes")
    if any(int(value) <= 0 for value in config.threshold_hidden):
        raise ValueError(f"threshold_hidden values must be positive, got {config.threshold_hidden}")
    if config.threshold_log_std_min > config.threshold_log_std_max:
        raise ValueError("threshold_log_std_min must be <= threshold_log_std_max")
    if not config.threshold_log_std_min <= config.threshold_log_std_initial <= config.threshold_log_std_max:
        raise ValueError(
            "threshold_log_std_initial must be within "
            f"[{config.threshold_log_std_min}, {config.threshold_log_std_max}], got {config.threshold_log_std_initial}"
        )
    if config.threshold_entropy_coef < 0.0:
        raise ValueError(f"threshold_entropy_coef must be non-negative, got {config.threshold_entropy_coef}")
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
