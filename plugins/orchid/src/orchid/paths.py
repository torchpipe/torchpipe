from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = PLUGIN_ROOT / "benchmarks"
BENCHMARK_ARTIFACTS_DIR = BENCHMARKS_DIR / "artifacts"
BENCHMARK_DATASETS_DIR = BENCHMARKS_DIR / "datasets"
DOCS_DIR = PLUGIN_ROOT / "docs"
RESEARCH_DIR = PLUGIN_ROOT / "research"
TESTS_DIR = PLUGIN_ROOT / "tests"
TEST_ARTIFACTS_DIR = TESTS_DIR / "artifacts"


def plugin_path(*parts: str) -> str:
    return str(PLUGIN_ROOT.joinpath(*parts))


def benchmark_artifact(*parts: str) -> str:
    return str(BENCHMARK_ARTIFACTS_DIR.joinpath(*parts))


def benchmark_dataset(*parts: str) -> str:
    return str(BENCHMARK_DATASETS_DIR.joinpath(*parts))


def test_artifact(*parts: str) -> str:
    return str(TEST_ARTIFACTS_DIR.joinpath(*parts))
