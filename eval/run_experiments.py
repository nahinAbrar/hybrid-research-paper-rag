"""
Script for running evaluation experiments across different RAG configurations.
"""

def run_evaluation_suite(dataset_path: str, output_log_path: str) -> None:
    """
    Runs the full evaluation suite over a dataset of QA pairs and logs results.
    Compares configurations like BM25 only, Vector only, and Hybrid.

    Args:
        dataset_path: Path to the evaluation dataset (CSV or JSON).
        output_log_path: Path to save the evaluation results and metrics.
    """
    # TODO: implement
    raise NotImplementedError("Experiment runner is not yet implemented.")

if __name__ == "__main__":
    # TODO: implement CLI arguments and execution
    pass
