from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionIOConfig:
    """Shared collection I/O settings used across agents and orchestrator."""

    flush_every: int = 1
    async_writer: bool = False
    queue_size: int = 256
    compress_heavy: bool = False
    include_raw_model_output: bool = False


def resolve_io_config(
    io_config: CollectionIOConfig | None,
    *,
    writer_flush_every: int,
    writer_async: bool,
    writer_queue_size: int,
    compress_heavy: bool,
    include_raw_model_output: bool,
) -> CollectionIOConfig:
    if io_config is not None:
        return io_config
    return CollectionIOConfig(
        flush_every=writer_flush_every,
        async_writer=writer_async,
        queue_size=writer_queue_size,
        compress_heavy=compress_heavy,
        include_raw_model_output=include_raw_model_output,
    )
