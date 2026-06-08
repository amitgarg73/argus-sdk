from argus.session import TraceLogger
from argus.evals import write_eval
from argus.judge import evaluate_session_outputs
from argus.engine import (
    EvalResult,
    Incident,
    run_all_evals,
    run_all_detectors,
    run_quality_detectors,
    run_evals_and_persist,
    run_detectors_and_persist,
    compute_shadow_cb_fires,
    build_annotated_call_stack,
    generate_fix_suggestion,
    summarize_incident,
    load_pipeline_config,
)

__all__ = [
    # logging + L4/L5 evals
    "TraceLogger",
    "write_eval",
    "evaluate_session_outputs",
    # L3 operational evals
    "EvalResult",
    "run_all_evals",
    "run_evals_and_persist",
    # pattern detector
    "Incident",
    "run_all_detectors",
    "run_quality_detectors",
    "compute_shadow_cb_fires",
    "run_detectors_and_persist",
    # RCA
    "build_annotated_call_stack",
    "generate_fix_suggestion",
    "summarize_incident",
    # config loader
    "load_pipeline_config",
]
__version__ = "0.3.0"
