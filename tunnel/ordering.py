from __future__ import annotations

from dataclasses import dataclass

from .model import LocalFlowModel, TrackFeatures
from .types import Track


@dataclass(frozen=True)
class OrderingConfig:
    seed_count: int = 8
    swap_passes: int = 3
    swap_limit: int = 220


@dataclass(frozen=True)
class OrderedPlaylist:
    tracks: list[Track]
    score: float
    model_name: str
    missing_bpm: int
    missing_year: int
    local_files: int
    audio_features: int
    embeddings: int


def order_tracks(
    tracks: list[Track],
    model: LocalFlowModel | None = None,
    config: OrderingConfig | None = None,
) -> OrderedPlaylist:
    if model is None:
        model = LocalFlowModel()
    if config is None:
        config = OrderingConfig()
    if not tracks:
        return OrderedPlaylist([], 0.0, model.name, 0, 0, 0, model.audio_feature_count, model.embedding_count)
    if len(tracks) == 1:
        return OrderedPlaylist(
            tracks[:],
            0.0,
            model.name,
            int(tracks[0].bpm is None),
            int(tracks[0].year is None),
            int(bool(tracks[0].location)),
            model.audio_feature_count,
            model.embedding_count,
        )

    features = [model.features_for(track) for track in tracks]
    transition_cache = _build_transition_cache(tracks, features, model)
    seed_indices = _seed_indices(features, model, len(tracks), config.seed_count)

    best_route: list[int] | None = None
    best_score = float("inf")
    for seed_index in seed_indices:
        route = _greedy_route(seed_index, features, model, transition_cache)
        route = _improve_by_swaps(route, features, model, transition_cache, config)
        score = _route_score(route, features, model, transition_cache)
        if score < best_score:
            best_route = route
            best_score = score

    assert best_route is not None
    ordered = [tracks[index] for index in best_route]
    return OrderedPlaylist(
        tracks=ordered,
        score=best_score,
        model_name=model.name,
        missing_bpm=sum(1 for track in tracks if track.bpm is None),
        missing_year=sum(1 for track in tracks if track.year is None),
        local_files=sum(1 for track in tracks if track.location),
        audio_features=model.audio_feature_count,
        embeddings=model.embedding_count,
    )


def _build_transition_cache(
    tracks: list[Track],
    features: list[TrackFeatures],
    model: LocalFlowModel,
) -> list[list[float]]:
    cache: list[list[float]] = [[0.0] * len(tracks) for _ in tracks]
    for left_index, left in enumerate(tracks):
        for right_index, right in enumerate(tracks):
            if left_index == right_index:
                continue
            cache[left_index][right_index] = model.transition(
                left,
                right,
                features[left_index],
                features[right_index],
            ).total
    return cache


def _seed_indices(
    features: list[TrackFeatures],
    model: LocalFlowModel,
    total: int,
    seed_count: int,
) -> list[int]:
    target = model.target_energy(0, total)
    ranked = sorted(
        range(total),
        key=lambda index: (abs(features[index].energy - target), features[index].energy),
    )
    return ranked[: max(1, min(seed_count, total))]


def _greedy_route(
    seed_index: int,
    features: list[TrackFeatures],
    model: LocalFlowModel,
    transition_cache: list[list[float]],
) -> list[int]:
    total = len(features)
    route = [seed_index]
    remaining = set(range(total))
    remaining.remove(seed_index)

    while remaining:
        position = len(route)
        previous = route[-1]
        recent = set(route[-2:])
        best_index = min(
            remaining,
            key=lambda candidate: (
                transition_cache[previous][candidate]
                + model.placement_cost(features[candidate], position, total)
                + _recent_repetition_cost(candidate, recent, transition_cache)
            ),
        )
        route.append(best_index)
        remaining.remove(best_index)

    return route


def _recent_repetition_cost(candidate: int, recent: set[int], transition_cache: list[list[float]]) -> float:
    if not recent:
        return 0.0
    close_repeats = sum(1 for index in recent if transition_cache[index][candidate] < 0.05)
    return 0.18 * close_repeats


def _improve_by_swaps(
    route: list[int],
    features: list[TrackFeatures],
    model: LocalFlowModel,
    transition_cache: list[list[float]],
    config: OrderingConfig,
) -> list[int]:
    if len(route) > config.swap_limit:
        return route

    improved_route = route[:]
    for _ in range(config.swap_passes):
        improved = False
        for left_position in range(len(improved_route) - 1):
            for right_position in range(left_position + 1, len(improved_route)):
                delta = _swap_delta(
                    improved_route,
                    left_position,
                    right_position,
                    features,
                    model,
                    transition_cache,
                )
                if delta < -1e-9:
                    improved_route[left_position], improved_route[right_position] = (
                        improved_route[right_position],
                        improved_route[left_position],
                    )
                    improved = True
        if not improved:
            break
    return improved_route


def _swap_delta(
    route: list[int],
    left_position: int,
    right_position: int,
    features: list[TrackFeatures],
    model: LocalFlowModel,
    transition_cache: list[list[float]],
) -> float:
    total = len(route)
    positions = {left_position, right_position}
    affected_edges = _affected_edges(left_position, right_position, total)

    old_score = sum(
        model.placement_cost(features[route[position]], position, total)
        for position in positions
    )
    old_score += sum(transition_cache[route[left]][route[right]] for left, right in affected_edges)

    swapped = route[:]
    swapped[left_position], swapped[right_position] = swapped[right_position], swapped[left_position]
    new_score = sum(
        model.placement_cost(features[swapped[position]], position, total)
        for position in positions
    )
    new_score += sum(transition_cache[swapped[left]][swapped[right]] for left, right in affected_edges)

    return new_score - old_score


def _affected_edges(left_position: int, right_position: int, total: int) -> set[tuple[int, int]]:
    positions = {left_position, right_position}
    edges: set[tuple[int, int]] = set()
    for position in positions:
        if position > 0:
            edges.add((position - 1, position))
        if position < total - 1:
            edges.add((position, position + 1))
    return edges


def _route_score(
    route: list[int],
    features: list[TrackFeatures],
    model: LocalFlowModel,
    transition_cache: list[list[float]],
) -> float:
    total = len(route)
    placement = sum(
        model.placement_cost(features[index], position, total)
        for position, index in enumerate(route)
    )
    transitions = sum(
        transition_cache[left][right]
        for left, right in zip(route, route[1:])
    )
    return placement + transitions
