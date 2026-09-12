# DB Pension Asset Liability Model

An interactive Streamlit prototype for modelling a defined benefit pension scheme alongside a stochastic investment portfolio.

## What it models

- Active members, annual new entrants, payroll and employee/employer contributions
- CARE benefits or an aggregate final salary approximation
- Retirement and indexed pension payments
- Age and sex specific mortality using the supplied ELT15, PMA92C and PFA92C tables
- Present value of benefits accrued to date for active members and pensioners
- Stochastic, annually rebalanced investment returns across seven asset classes
- Funding ratio distributions, probability of full funding and probability of asset exhaustion

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Deploy on Render

1. Put this project in a GitHub repository.
2. In Render, select **New > Blueprint** and connect the repository.
3. Render will use `render.yaml` to install the dependencies and start the app.

## Modelling conventions

Liabilities are calculated on an accrued benefits basis. For an active member cohort aged \(x\), the liability is the accrued annual pension multiplied by the probability of surviving to retirement, the pensioner annuity factor at retirement and the discount factor to retirement. Pensioner liabilities are the indexed life annuity value of pensions currently in payment.

Deaths can be simulated using binomial draws, while a path level mortality multiplier introduces systematic longevity uncertainty. The mortality improvement control applies a constant annual reduction to the supplied base mortality rates.

Investment returns use the allocation weighted mean and covariance implied by the asset class assumptions and the fixed correlation matrix in `model.py`. Returns are modelled as lognormal and the portfolio is assumed to rebalance annually.

## Important limitation

The supplied mortality workbook contains historic mortality tables. They are included faithfully for demonstration and sensitivity analysis, but are not appropriate as a current UK pension scheme funding or accounting valuation basis. A production version should use scheme specific experience and a current CMI mortality projection model, reviewed by a qualified actuary.
