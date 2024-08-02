"""Result collector approaches."""

from collections import defaultdict


def rrf(
    list_of_items: list[list[tuple[str, int]]], k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion.

    For each list, we calculate a score using on RRF formula based
    on it's position in the list. We assign a score for each text
    in the lists.
    The original paper uses k=60 for best results:
    https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

    Args:
        list_of_items (list[list[tuple[str, float]]]): List of texts to be ranked.
            Each list is a tuple of (text, score).
        k (int, optional): Constant for smoothing.
            Defaults to 60.

    Returns:
        list[tuple[str, int]]: List of reranked text and corresponding score.
    """
    fused_results: dict[str, float] = defaultdict(float)

    for result_list in list_of_items:
        for position, (text, _) in enumerate(result_list):
            fused_results[text] += 1.0 / (position + k)

    # Sort items based on their RRF scores in descending order
    sorted_items = dict(sorted(fused_results.items(), key=lambda x: x[1], reverse=True))
    return [(text, score) for text, score in sorted_items.items()]  # noqa: C416
