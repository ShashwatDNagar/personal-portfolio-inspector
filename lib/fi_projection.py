"""Monte Carlo FI projection, Coast FI calculator, and FI number computation."""

import numpy as np


def monte_carlo_projection(
    current_value: float,
    annual_contribution: float,
    years: int,
    n_simulations: int = 1000,
    mean_return: float = 0.10,
    std_return: float = 0.16,
) -> dict:
    """Simulate n_simulations portfolio growth paths."""
    rng = np.random.default_rng(42)
    paths = np.zeros((n_simulations, years + 1))
    paths[:, 0] = current_value

    for y in range(1, years + 1):
        returns = rng.normal(mean_return, std_return, n_simulations)
        paths[:, y] = paths[:, y - 1] * (1 + returns) + annual_contribution

    year_list = list(range(years + 1))
    return {
        "paths": paths,
        "median": np.median(paths, axis=0).tolist(),
        "p10": np.percentile(paths, 10, axis=0).tolist(),
        "p25": np.percentile(paths, 25, axis=0).tolist(),
        "p75": np.percentile(paths, 75, axis=0).tolist(),
        "p90": np.percentile(paths, 90, axis=0).tolist(),
        "years": year_list,
        "final_median": float(np.median(paths[:, -1])),
        "final_p10": float(np.percentile(paths[:, -1], 10)),
        "final_p90": float(np.percentile(paths[:, -1], 90)),
    }


def probability_of_fi(
    simulations_result: dict,
    fi_target: float,
    current_age: int | None = None,
) -> dict:
    """Compute what fraction of simulations reached the FI target."""
    paths = simulations_result["paths"]
    n_simulations = paths.shape[0]
    years = paths.shape[1] - 1

    hit_fi = np.any(paths >= fi_target, axis=1)
    probability = float(np.mean(hit_fi))

    fi_years = []
    for i in range(n_simulations):
        crossings = np.where(paths[i] >= fi_target)[0]
        if len(crossings) > 0:
            fi_years.append(int(crossings[0]))

    median_years = float(np.median(fi_years)) if fi_years else None
    earliest = int(min(fi_years)) if fi_years else None
    latest = int(max(fi_years)) if fi_years else None

    age_str = ""
    if current_age is not None and median_years is not None:
        fi_age = current_age + int(median_years)
        age_str = f" (age {fi_age})"

    if probability >= 0.9:
        desc = (
            f"You have a {probability*100:.0f}% chance of reaching your FI target of "
            f"${fi_target:,.0f}. In the median case, you'll get there in about "
            f"{median_years:.0f} years{age_str}. You're in great shape."
        )
    elif probability >= 0.5:
        desc = (
            f"You have a {probability*100:.0f}% chance of reaching ${fi_target:,.0f}. "
            f"The median path gets there in ~{median_years:.0f} years{age_str}. "
            f"Increasing contributions would improve your odds."
        )
    elif probability > 0:
        desc = (
            f"Only {probability*100:.0f}% of simulated scenarios reach ${fi_target:,.0f} "
            f"within {years} years. Consider increasing contributions or extending your timeline."
        )
    else:
        desc = (
            f"None of the {n_simulations} simulated scenarios reached ${fi_target:,.0f} "
            f"in {years} years. You may need to increase contributions significantly or "
            f"extend your timeline."
        )

    return {
        "probability": probability,
        "median_years_to_fi": median_years,
        "earliest_fi_year": earliest,
        "latest_fi_year": latest,
        "description": desc,
    }


def coast_fi(
    current_value: float,
    fi_target: float,
    current_age: int,
    fi_age: int,
    growth_rate: float = 0.08,
) -> dict:
    """Calculate Coast FI: when can you stop contributing and still hit FI?"""
    years_remaining = fi_age - current_age
    if years_remaining <= 0:
        return {
            "coast_fi_number": fi_target,
            "have_enough_to_coast": current_value >= fi_target,
            "surplus_or_deficit": current_value - fi_target,
            "coast_fi_age": current_age if current_value >= fi_target else None,
            "description": "You've reached your target FI age.",
        }

    coast_fi_number = fi_target / (1 + growth_rate) ** years_remaining
    have_enough = current_value >= coast_fi_number
    surplus_or_deficit = current_value - coast_fi_number

    coast_fi_age = None
    for y in range(100):
        if current_value * (1 + growth_rate) ** y >= fi_target:
            coast_fi_age = current_age + y
            break

    if have_enough:
        desc = (
            f"You've reached Coast FI! Your portfolio (${current_value:,.0f}) is already above "
            f"the ${coast_fi_number:,.0f} needed to coast to ${fi_target:,.0f} by age {fi_age}. "
            f"Even if you stopped contributing today, your investments would grow to your FI target "
            f"at an average 8% return."
        )
        if coast_fi_age and coast_fi_age != fi_age:
            desc += f" Without contributions, you'd hit FI around age {coast_fi_age}."
    else:
        desc = (
            f"You need ${coast_fi_number:,.0f} to coast to FI — you're ${abs(surplus_or_deficit):,.0f} short. "
            f"Keep contributing until your portfolio reaches that number, then you could theoretically "
            f"stop and let compound growth do the rest."
        )

    return {
        "coast_fi_number": coast_fi_number,
        "have_enough_to_coast": have_enough,
        "surplus_or_deficit": surplus_or_deficit,
        "coast_fi_age": coast_fi_age,
        "description": desc,
    }


def compute_fi_number(
    monthly_expenses: float | None = None,
    annual_income: float = 150000,
    withdrawal_rate: float = 0.04,
) -> float:
    """Compute FI target number from expenses or income."""
    if monthly_expenses is not None and monthly_expenses > 0:
        return (monthly_expenses * 12) / withdrawal_rate
    return (annual_income * 0.6) / withdrawal_rate
