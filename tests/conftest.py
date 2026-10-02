import pytest

from research_pilot.config import Settings


@pytest.fixture
def mock_settings(tmp_path):
    def make(**overrides):
        values = dict(
            _env_file=None,
            llm_provider="mock",
            llm_model="mock",
            literature_providers="mock",
            experiment_executor="simulated",
            workspace_dir=str(tmp_path / "workspace"),
        )
        values.update(overrides)
        return Settings(**values)

    return make
