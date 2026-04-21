from src.evaluation.s_alpaca_eval import AlpacaEval


class MTBench(AlpacaEval):
    """MT-Bench shares the same win-rate evaluation logic as AlpacaEval.
    Multi-turn context is pre-flattened into the 'prompt' field by the pipeline."""
    pass
