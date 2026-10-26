import pandas as pd
import pytest

from src.generate_data import generate


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    return generate(n_rows=600, seed=7)
