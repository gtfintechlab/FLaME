"""FLAME tasks for BenchForge.

This module provides implementations of FLAME benchmark tasks
with full BenchForge integration.
"""

from bench_forge.flame.tasks.fomc import FOMCConfig, FOMCTask, register_fomc_task
from bench_forge.flame.tasks.convfinqa import ConvFinQAConfig, ConvFinQATask
from bench_forge.flame.tasks.finqa import FinQAConfig, FinQATask
from bench_forge.flame.tasks.finer import FiNERConfig, FiNERTask
from bench_forge.flame.tasks.finentity import FinEntityConfig, FinEntityTask
from bench_forge.flame.tasks.edtsum import EDTSumConfig, EDTSumTask
from bench_forge.flame.tasks.causal_detection import (
    CausalDetectionConfig,
    CausalDetectionTask,
    register_causal_detection_task,
)
from bench_forge.flame.tasks.causal_classification import (
    CausalClassificationConfig,
    CausalClassificationTask,
    register_causal_classification_task,
)
from bench_forge.flame.tasks.numclaim import (
    NumClaimConfig,
    NumClaimTask,
    register_numclaim_task,
)
from bench_forge.flame.tasks.tatqa import TATQAConfig, TATQATask
from bench_forge.flame.tasks.banking77 import Banking77Config, Banking77Task
from bench_forge.flame.tasks.ectsum import ECTSumConfig, ECTSumTask
from bench_forge.flame.tasks.finbench import FinBenchConfig, FinBenchTask
from bench_forge.flame.tasks.fiqa_sa import FiQASAConfig, FiQASATask

__all__ = [
    "FOMCTask",
    "FOMCConfig",
    "register_fomc_task",
    "ConvFinQATask",
    "ConvFinQAConfig",
    "FinQATask",
    "FinQAConfig",
    "FiNERTask",
    "FiNERConfig",
    "FinEntityTask",
    "FinEntityConfig",
    "EDTSumTask",
    "EDTSumConfig",
    "CausalDetectionTask",
    "CausalDetectionConfig",
    "register_causal_detection_task",
    "CausalClassificationTask",
    "CausalClassificationConfig",
    "register_causal_classification_task",
    "NumClaimTask",
    "NumClaimConfig",
    "register_numclaim_task",
    "TATQATask",
    "TATQAConfig",
    "Banking77Task",
    "Banking77Config",
    "ECTSumTask",
    "ECTSumConfig",
    "FinBenchTask",
    "FinBenchConfig",
    "FiQASATask",
    "FiQASAConfig",
]


# Auto-register all FLAME tasks
def register_all_flame_tasks():
    """Register all FLAME tasks with BenchForge."""
    register_fomc_task()
    register_causal_detection_task()
    register_causal_classification_task()
    register_numclaim_task()

    # QA tasks are auto-registered via @flame_task decorator
    # but we can import them to ensure registration
    from bench_forge.flame.tasks import (
        convfinqa,  # noqa: F401
        finqa,  # noqa: F401
        edtsum,  # noqa: F401
        fiqa_sa,  # noqa: F401
        tatqa,  # noqa: F401
        banking77,  # noqa: F401
        ectsum,  # noqa: F401
        finbench,  # QA and classification tasks  # noqa: F401
        finer,  # noqa: F401
        finentity,  # NER tasks  # noqa: F401
    )
