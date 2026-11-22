"""Per-million-token prices (USD) used ONLY for cost estimates in evaluation output.

Source: Anthropic's models overview page, read on 2026-11-10. Prices change; verify before relying on
them. Haiku 5.5 is listed as "from" these figures, so its estimate is a lower bound. Open models
run on your own hardware and are priced at zero here.
"""
PRICE_PER_MTOK = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-haiku-5-5": (0.10, 0.50),   # "from" price: lower bound
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int):
    """Return estimated USD, or None if the model's price is unknown. Open models cost 0."""
    if model.startswith("ollama:"):
        return 0.0
    if model not in PRICE_PER_MTOK:
        return None
    pin, pout = PRICE_PER_MTOK[model]
    return (input_tokens * pin + output_tokens * pout) / 1_000_000