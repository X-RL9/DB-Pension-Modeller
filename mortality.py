from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl


@dataclass
class MortalityBasis:
    """Mortality rates loaded from the user's 2002-era workbook."""

    rates: dict[str, dict[int, float]]
    base_year: int = 2000

    @classmethod
    def from_workbook(cls, path: str | Path) -> "MortalityBasis":
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        sheet_map = {
            "male_active": "ELT15 Males",
            "female_active": "ELT15 Females",
            "male_pensioner": "PMA92C",
            "female_pensioner": "PFA92C",
        }
        rates: dict[str, dict[int, float]] = {}
        for key, sheet_name in sheet_map.items():
            ws = wb[sheet_name]
            table: dict[int, float] = {}
            expected_age: int | None = None
            for row in ws.iter_rows(min_row=4, values_only=True):
                candidates = []
                if len(row) >= 4:
                    candidates.append((row[0], row[3]))
                if len(row) >= 5:
                    candidates.append((row[1], row[4]))
                valid = [
                    (int(age), float(qx))
                    for age, qx in candidates
                    if isinstance(age, (int, float))
                    and isinstance(qx, (int, float))
                    and 0.0 <= float(qx) <= 1.0
                ]
                if not valid:
                    continue
                if expected_age is None:
                    age, qx = valid[0]
                else:
                    matched = [item for item in valid if item[0] == expected_age]
                    if not matched:
                        continue
                    age, qx = matched[0]
                table[age] = qx
                expected_age = age + 1
            rates[key] = table
        return cls(rates=rates)

    def base_qx(self, sex: int, age: int, status: str) -> float:
        sex_name = "male" if sex == 0 else "female"
        key = f"{sex_name}_{status}"
        table = self.rates[key]

        # Pensioner tables begin at 50. Population rates bridge younger ages.
        if age not in table and status == "pensioner":
            active_table = self.rates[f"{sex_name}_active"]
            if age in active_table:
                return active_table[age]

        if age in table:
            return table[age]

        min_age, max_age = min(table), max(table)
        if age < min_age:
            return table[min_age]

        # Smoothly extend the pensioner tables beyond age 105 and close the
        # table at age 120. This avoids assuming survival beyond the table.
        last_q = table[max_age]
        extended = last_q * (1.15 ** (age - max_age))
        if age >= 120:
            return 1.0
        return min(0.999, extended)

    def qx(
        self,
        sex: int,
        age: int,
        calendar_year: int,
        status: str,
        annual_improvement: float,
    ) -> float:
        if age >= 120:
            return 1.0
        base = self.base_qx(sex, age, status)
        years = calendar_year - self.base_year
        adjusted = base * ((1.0 - annual_improvement) ** years)
        return float(np.clip(adjusted, 0.0, 1.0))


def mortality_metadata() -> dict[str, str]:
    return {
        "active_males": "ELT15 Males",
        "active_females": "ELT15 Females",
        "pensioner_males": "PMA92C projected to 2000",
        "pensioner_females": "PFA92C projected to 2000",
        "warning": (
            "Historic tables supplied by the user. Suitable for demonstration "
            "and sensitivity testing, not a current scheme valuation basis."
        ),
    }
