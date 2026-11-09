"""Cost per call, from token counts. Prices are US dollars per million tokens."""

from typing import Any

# input, output, cache read, cache write (5-minute TTL = 1.25x input)
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    # The evaluation judge (prompts under 100K tokens).
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
    # Possible refusal-fallback targets.
    "claude-opus-5": (5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": (5.00, 25.00, 0.50, 6.25),
}


def _cost(u: dict[str, Any], model: str | None) -> float | None:
    prices = PRICES.get(model or "")
    if prices is None:
        return None
    p_in, p_out, p_read, p_write = prices
    return (
        (u.get("input_tokens") or 0) * p_in
        + (u.get("output_tokens") or 0) * p_out
        + (u.get("cache_read_input_tokens") or 0) * p_read
        + (u.get("cache_creation_input_tokens") or 0) * p_write
    ) / 1_000_000


def cost_usd(usage: dict[str, Any], served_model: str | None) -> float | None:
    """Total cost of one call. After a refusal fallback, `usage.iterations` lists every
    attempt; each is priced at the rates of the model that ran it."""
    iterations = usage.get("iterations") or []
    if not iterations:
        return _cost(usage, served_model)
    total = 0.0
    for it in iterations:
        cost = _cost(it, it.get("model") or served_model)
        if cost is None:
            return None
        total += cost
    return total
