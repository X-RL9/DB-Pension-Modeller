from pathlib import Path

import numpy as np

from model import DEFAULT_PORTFOLIO, SchemeConfig, portfolio_statistics, simulate
from mortality import MortalityBasis


ROOT = Path(__file__).resolve().parent


def main() -> None:
    basis = MortalityBasis.from_workbook(ROOT / "data" / "2002_mortality_tables_single_lives.xlsx")
    assert 0 < basis.base_qx(0, 60, "active") < 1
    assert 0 < basis.base_qx(1, 60, "active") < 1
    config = SchemeConfig(projection_years=5, simulations=30, seed=7)
    result = simulate(config, basis, DEFAULT_PORTFOLIO)
    summary = result["summary"]
    _, expected_return, volatility = portfolio_statistics(DEFAULT_PORTFOLIO)

    assert len(result["percentiles"]) == 6
    assert np.isfinite(result["history"]["liabilities"]).all()
    assert (result["history"]["liabilities"] > 0).all()
    assert (result["history"]["assets"] >= 0).all()
    assert 0 < expected_return < 0.2
    assert 0 < volatility < 0.3
    assert 0 <= summary["fully_funded_probability"] <= 1
    assert 0 <= summary["asset_exhaustion_probability"] <= 1
    print("Model smoke test passed")
    print(summary)


if __name__ == "__main__":
    main()
