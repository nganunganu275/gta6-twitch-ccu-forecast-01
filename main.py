"""Forecast GTA VI launch-day Twitch viewers, hour by hour.

The program reads one case (makefun/twitch-ccu-case/v1): the Twitch viewer count
of a game's category up to the end of its release day's first hour, that hour's
count included, plus the release-day curves of past big launches. It writes, for
each of hours 2 to 24, five quantiles (5%, 25%, 50%, 75%, 95%) of that hour's
count.

Model: for each forecast hour separately, a locally weighted linear regression
of the log ratio ln(viewers_h / viewers_1) on ln(viewers_1), fitted on the
case's comparables. The shape of a launch day depends strongly on how big the
first hour already is: across past launches the evening log ratio falls almost
one for one as the hour-1 print grows (a launch whose first hour is already
huge has little room left to climb, while a quiet first hour leaves a steep
evening ramp), so the regression's slope carries that regularity instead of a
single fixed shape. Comparables are weighted by a Gaussian kernel on
ln(viewers_1), so launches of a similar size to this case count most. The
quantile spread is the fit's weighted residual spread, slightly shrunk, with a
floor: hours the comparables disagree about get wide bands, hours they agree
on get narrow ones. The spread is a prediction interval, so a target whose
hour-1 print sits far from the comparables the fit leans on earns a wider
band still, through the usual leverage term.

A comparable trains the model only when its release day is well covered and its
first hour already counts real launch viewers: a tiny first hour means that
game had not unlocked yet at midnight UTC, so its ratios describe a different
regime. With too few usable comparables the forecast falls back to persistence
(every hour at the first hour's count) with a fixed spread.

Standard library only, no network, deterministic: the same case always writes
the same bytes.
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# A comparable only trains the model when its first hour already counts real
# launch viewers and most of its release day is present.
MIN_HOUR1_VIEWERS = 10000
MIN_RELEASE_DAY_HOURS = 18
# Gaussian kernel width, in log viewers, for weighting comparables by how close
# their hour-1 print is to this case's.
KERNEL_WIDTH = 1.5
# The quantile spread is the fit's residual spread, slightly shrunk, with a
# floor so no hour is ever forecast with false certainty.
SPREAD_SHRINK = 0.9
SPREAD_FLOOR = 0.15
# Fallback spread when the case carries too few usable comparables to fit.
FALLBACK_SPREAD = 0.35
MIN_TRAINING_ROWS = 5
MAX_VIEWERS = 1_000_000_000


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up. A whole number comes back as an int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def release_day_curve(comparable: dict) -> list:
    """The comparable's 24 hourly viewer counts on its release day, hours 1..24.

    Timestamps are ISO 8601 UTC ("YYYY-MM-DDTHH:MM:SSZ"), so the release day's
    date and each point's hour come from fixed positions in the string. A
    missing or non-positive point stays None.
    """
    day = comparable["day_start"][:10]
    hours: list = [None] * 24
    for point in comparable.get("hourly") or []:
        stamp = point.get("from") or ""
        if len(stamp) < 13 or stamp[10] != "T" or stamp[:10] != day:
            continue
        viewers = point.get("viewers")
        if isinstance(viewers, (int, float)) and viewers > 0:
            hours[int(stamp[11:13])] = float(viewers)
    return hours


def training_rows(case: dict) -> list:
    """One row per usable comparable: (log hour-1 viewers, [log ratio per hour]).

    Each row's ratios are ln(viewers_h / viewers_1) for hours 2..24, None where
    that hour has no point.
    """
    rows = []
    for comparable in case.get("comparables") or []:
        curve = release_day_curve(comparable)
        first = curve[0]
        if first is None or first < MIN_HOUR1_VIEWERS:
            continue
        if sum(1 for v in curve if v is not None) < MIN_RELEASE_DAY_HOURS:
            continue
        log_first = math.log(first)
        ratios = [
            math.log(viewers) - log_first if viewers is not None else None
            for viewers in curve[1:]
        ]
        rows.append((log_first, ratios))
    return rows


def fit_hour(points: list, target_log_first: float):
    """Weighted least squares of one hour's log ratio on log hour-1 viewers.

    points is [(log hour-1 viewers, log ratio)] for this hour. Returns
    (intercept, slope, spread), or None when the points carry no weight.
    """
    weighted = []
    for log_first, ratio in points:
        weight = math.exp(-(((log_first - target_log_first) / KERNEL_WIDTH) ** 2))
        weighted.append((weight, log_first, ratio))
    total_weight = sum(w for w, _, _ in weighted)
    if total_weight <= 0:
        return None
    mean_x = sum(w * x for w, x, _ in weighted) / total_weight
    mean_y = sum(w * y for w, _, y in weighted) / total_weight
    sxx = sum(w * (x - mean_x) ** 2 for w, x, _ in weighted)
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in weighted)
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    intercept = mean_y - slope * mean_x
    variance = (
        sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in weighted)
        / total_weight
    )
    # A prediction interval, not just the residual spread: a target far from
    # the comparables the fit leans on earns a wider band. n_eff is the
    # effective number of comparables after kernel weighting; the leverage term
    # is the target's squared distance from the weighted centre, in units of
    # the weighted spread of the training points.
    sum_sq_weights = sum(w * w for w, _, _ in weighted)
    n_eff = (
        total_weight * total_weight / sum_sq_weights if sum_sq_weights > 0 else 0.0
    )
    leverage = ((target_log_first - mean_x) ** 2 / sxx) if sxx > 1e-9 else 0.0
    prediction_variance = variance * (1 + 1 / max(1.0, n_eff) + leverage)
    spread = max(SPREAD_FLOOR, SPREAD_SHRINK * math.sqrt(prediction_variance))
    return (intercept, slope, spread)


def hour_quantiles(p1: float, fit, target_log_first: float) -> list:
    """The five quantiles of one hour: median ratio from the fit, spread around it."""
    if fit is None:
        median_log_ratio, spread = 0.0, FALLBACK_SPREAD
    else:
        intercept, slope, spread = fit
        median_log_ratio = intercept + slope * target_log_first
    quantiles: list = []
    previous = 0.0
    for z in Z:
        value = p1 * math.exp(median_log_ratio + z * spread)
        value = min(max(value, 1.0), MAX_VIEWERS)
        value = round2(value)
        value = max(value, previous)  # rounding must never break the ordering
        quantiles.append(value)
        previous = value
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    target_log_first = math.log(p1)
    rows = training_rows(case)
    fits = {}
    if len(rows) >= MIN_TRAINING_ROWS:
        for position, hour in enumerate(range(2, 25)):
            points = [
                (log_first, ratios[position])
                for log_first, ratios in rows
                if ratios[position] is not None
            ]
            if len(points) >= MIN_TRAINING_ROWS:
                fits[hour] = fit_hour(points, target_log_first)
    hours = [
        {
            "hour": row["hour"],
            "q": hour_quantiles(p1, fits.get(row["hour"]), target_log_first),
        }
        for row in case["forecast"]
    ]
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
