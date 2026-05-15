from .paths import BENCHMARK_ARTIFACTS_DIR, BENCHMARK_DATASETS_DIR, DOCS_DIR, PLUGIN_ROOT, RESEARCH_DIR, TEST_ARTIFACTS_DIR


def __getattr__(name: str):
    if name == "ContinuousBatchingEngine":
        from .llmscheduler.core.engine import ContinuousBatchingEngine

        return ContinuousBatchingEngine
    if name == "TensorRTOfflineEngine":
        from .llmscheduler.engine.offline_engine import TensorRTOfflineEngine

        return TensorRTOfflineEngine
    if name == "TensorRTModelRuntime":
        from .llmscheduler.runtime.trt_runtime import TensorRTModelRuntime

        return TensorRTModelRuntime
    if name == "create_app":
        from .llmscheduler.server.app_factory import create_app

        return create_app
    raise AttributeError(name)

__all__ = [
    "BENCHMARK_ARTIFACTS_DIR",
    "BENCHMARK_DATASETS_DIR",
    "ContinuousBatchingEngine",
    "DOCS_DIR",
    "PLUGIN_ROOT",
    "RESEARCH_DIR",
    "TEST_ARTIFACTS_DIR",
    "TensorRTModelRuntime",
    "TensorRTOfflineEngine",
    "create_app",
]
