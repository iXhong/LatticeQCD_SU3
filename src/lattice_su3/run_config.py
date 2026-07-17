"""TOML-backed run configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class UpdateConfig:
    """Update algorithm settings.

    Inputs:
        backend: Heatbath backend name.
        overrelaxation_sweeps: Number of overrelaxation sweeps after heatbath.
        algorithm: Update algorithm name.
        step_size: Metropolis proposal step size.
    Outputs:
        Immutable update configuration.
    """

    backend: str = "jit_checkerboard"
    overrelaxation_sweeps: int = 2
    algorithm: str = "heatbath"
    step_size: float = 0.4


@dataclass(frozen=True)
class MeasureConfig:
    """Online measurement settings.

    Inputs:
        plaquette_every: Sweep interval for average plaquette measurements.
    Outputs:
        Immutable measurement configuration.
    """

    plaquette_every: int = 10


@dataclass(frozen=True)
class SaveConfig:
    """Run output settings.

    Inputs:
        config_every: Sweep interval for saved gauge configurations.
        overwrite: Whether existing output files may be replaced.
    Outputs:
        Immutable save configuration.
    """

    config_every: int = 10
    overwrite: bool = False


@dataclass(frozen=True)
class ThermalizeConfig:
    """Thermalization workflow settings.

    Inputs:
        name: Run directory name under results/runs.
        shape: Lattice shape.
        beta: Wilson gauge coupling.
        sweeps: Number of thermalization sweeps.
        seed: Random seed.
        start: Initial condition, hot or cold.
        update: Update settings.
        measure: Online measurement settings.
        save: Output settings.
    Outputs:
        Immutable thermalization configuration.
    """

    name: str
    shape: tuple[int, ...]
    beta: float
    sweeps: int
    seed: int
    start: str
    update: UpdateConfig
    measure: MeasureConfig
    save: SaveConfig


@dataclass(frozen=True)
class EnsembleConfig:
    """Production ensemble workflow settings.

    Inputs:
        name: Run directory name under results/runs.
        shape: Lattice shape.
        beta: Wilson gauge coupling.
        source_config: Thermalized source configuration path.
        chains: Number of independent production chains.
        sweeps_per_chain: Number of sweeps per chain.
        discard_sweeps: Initial production sweeps to omit from saving.
        seed_base: Base seed; chain index is added to this value.
        parallel: Maximum concurrent chains.
        update: Update settings.
        measure: Online measurement settings.
        save: Output settings.
    Outputs:
        Immutable ensemble configuration.
    """

    name: str
    shape: tuple[int, ...]
    beta: float
    source_config: Path
    chains: int
    sweeps_per_chain: int
    discard_sweeps: int
    seed_base: int
    parallel: int
    update: UpdateConfig
    measure: MeasureConfig
    save: SaveConfig


def load_toml(path: Path | str) -> dict[str, object]:
    """Load a TOML file.

    Inputs:
        path: TOML file path.
    Outputs:
        Parsed TOML mapping.
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


def _table(data: dict[str, object], name: str) -> dict[str, object]:
    """Read a TOML table as a dictionary.

    Inputs:
        data: Parsed TOML mapping.
        name: Table name.
    Outputs:
        Table dictionary, or an empty dictionary when absent.
    """
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _required(table: dict[str, object], key: str, table_name: str) -> object:
    """Read a required table value.

    Inputs:
        table: TOML table dictionary.
        key: Required key.
        table_name: Human-readable table name.
    Outputs:
        Value associated with the key.
    """
    if key not in table:
        raise ValueError(f"[{table_name}] is missing required key: {key}")
    return table[key]


def _shape(value: object) -> tuple[int, ...]:
    """Parse and validate a lattice shape.

    Inputs:
        value: TOML shape value.
    Outputs:
        Tuple of positive lattice extents.
    """
    if not isinstance(value, list):
        raise ValueError("shape must be a list of integer extents")
    shape = tuple(int(length) for length in value)
    if len(shape) < 2 or any(length <= 0 for length in shape):
        raise ValueError("shape must contain at least two positive extents")
    return shape


def _update_config(data: dict[str, object]) -> UpdateConfig:
    """Build update settings from parsed TOML data.

    Inputs:
        data: Parsed TOML mapping.
    Outputs:
        Update configuration.
    """
    table = _table(data, "update")
    config = UpdateConfig(
        backend=str(table.get("backend", "jit_checkerboard")),
        overrelaxation_sweeps=int(table.get("overrelaxation_sweeps", 2)),
        algorithm=str(table.get("algorithm", "heatbath")),
        step_size=float(table.get("step_size", 0.4)),
    )
    if config.algorithm != "heatbath":
        raise ValueError("only heatbath is supported by the new production workflow")
    if config.backend not in {"jit", "jit_checkerboard", "numpy"}:
        raise ValueError("backend must be 'jit', 'jit_checkerboard', or 'numpy'")
    if config.overrelaxation_sweeps < 0:
        raise ValueError("overrelaxation_sweeps must be non-negative")
    if config.step_size < 0.0:
        raise ValueError("step_size must be non-negative")
    return config


def _measure_config(data: dict[str, object]) -> MeasureConfig:
    """Build measurement settings from parsed TOML data.

    Inputs:
        data: Parsed TOML mapping.
    Outputs:
        Measurement configuration.
    """
    table = _table(data, "measure")
    config = MeasureConfig(plaquette_every=int(table.get("plaquette_every", 10)))
    if config.plaquette_every < 0:
        raise ValueError("plaquette_every must be non-negative")
    return config


def _save_config(data: dict[str, object]) -> SaveConfig:
    """Build save settings from parsed TOML data.

    Inputs:
        data: Parsed TOML mapping.
    Outputs:
        Save configuration.
    """
    table = _table(data, "save")
    config = SaveConfig(
        config_every=int(table.get("config_every", 10)),
        overwrite=bool(table.get("overwrite", False)),
    )
    if config.config_every < 0:
        raise ValueError("config_every must be non-negative")
    return config


def load_thermalize_config(path: Path | str) -> ThermalizeConfig:
    """Load a thermalization workflow configuration.

    Inputs:
        path: TOML file path.
    Outputs:
        Thermalization configuration.
    """
    data = load_toml(path)
    run = _table(data, "run")
    start = str(run.get("start", "hot"))
    config = ThermalizeConfig(
        name=str(_required(run, "name", "run")),
        shape=_shape(_required(run, "shape", "run")),
        beta=float(_required(run, "beta", "run")),
        sweeps=int(_required(run, "sweeps", "run")),
        seed=int(_required(run, "seed", "run")),
        start=start,
        update=_update_config(data),
        measure=_measure_config(data),
        save=_save_config(data),
    )
    if config.beta < 0.0:
        raise ValueError("beta must be non-negative")
    if config.sweeps < 0:
        raise ValueError("sweeps must be non-negative")
    if config.start not in {"hot", "cold"}:
        raise ValueError("thermalize start must be 'hot' or 'cold'")
    return config


def load_ensemble_config(path: Path | str) -> EnsembleConfig:
    """Load a production ensemble workflow configuration.

    Inputs:
        path: TOML file path.
    Outputs:
        Ensemble configuration.
    """
    data = load_toml(path)
    run = _table(data, "run")
    source = _table(data, "source")
    ensemble = _table(data, "ensemble")
    config = EnsembleConfig(
        name=str(_required(run, "name", "run")),
        shape=_shape(_required(run, "shape", "run")),
        beta=float(_required(run, "beta", "run")),
        source_config=Path(str(_required(source, "config", "source"))),
        chains=int(ensemble.get("chains", 4)),
        sweeps_per_chain=int(_required(ensemble, "sweeps_per_chain", "ensemble")),
        discard_sweeps=int(ensemble.get("discard_sweeps", 0)),
        seed_base=int(ensemble.get("seed_base", 12345)),
        parallel=int(ensemble.get("parallel", 1)),
        update=_update_config(data),
        measure=_measure_config(data),
        save=_save_config(data),
    )
    if config.beta < 0.0:
        raise ValueError("beta must be non-negative")
    if config.chains <= 0:
        raise ValueError("chains must be positive")
    if config.sweeps_per_chain < 0:
        raise ValueError("sweeps_per_chain must be non-negative")
    if config.discard_sweeps < 0:
        raise ValueError("discard_sweeps must be non-negative")
    if config.discard_sweeps > config.sweeps_per_chain:
        raise ValueError("discard_sweeps cannot exceed sweeps_per_chain")
    if config.parallel <= 0:
        raise ValueError("parallel must be positive")
    return config
