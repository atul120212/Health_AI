"""
Disease Surveillance AI — outbreak detection and early warning.

Algorithm:
  1. Pull weekly case counts from HMIS/IDSP for a district+disease pair.
  2. Compute a rolling baseline (mean + std) over the past N weeks,
     excluding the most recent 3 weeks so emerging outbreaks don't
     inflate the baseline.
  3. Calculate z-score for each recent week.
  4. If z-score ≥ OUTBREAK_ALERT_THRESHOLD for any of the last 3 weeks
     → declare an outbreak alert.
  5. Claude generates a plain-language alert narrative for the DHO,
     including trend analysis and recommended response actions.

Early-warning horizon: 2–3 weeks by monitoring the rising slope before
the z-score threshold is crossed (slope-based pre-alert).
"""

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from openai import AsyncOpenAI

from config.settings import (
    SARVAM_API_KEY,
    SARVAM_BASE_URL,
    SARVAM_LLM_MODEL,
    OUTBREAK_ALERT_THRESHOLD,
    SURVEILLANCE_LOOKBACK_DAYS,
    EARLY_WARNING_WEEKS,
)
from src.middleware import hmis_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def compute_baseline(
    series: pd.Series,
    exclude_recent_weeks: int = 3,
) -> tuple[float, float]:
    """
    Compute baseline mean and std from historical case counts,
    excluding the most recent weeks to avoid masking active outbreaks.
    """
    baseline = series.iloc[:-exclude_recent_weeks] if len(series) > exclude_recent_weeks else series
    mean = float(baseline.mean())
    std = float(baseline.std(ddof=1)) if len(baseline) > 1 else 1.0
    return mean, max(std, 0.1)  # floor std to avoid division by zero


def zscore_series(series: pd.Series, mean: float, std: float) -> pd.Series:
    return (series - mean) / std


def detect_slope_pre_alert(
    zscores: pd.Series,
    window: int = 4,
    slope_threshold: float = 0.4,
) -> bool:
    """
    Early warning: detect a consistently rising z-score slope over the
    last `window` weeks, even if threshold not yet crossed.
    """
    if len(zscores) < window:
        return False
    recent = zscores.iloc[-window:].values
    x = np.arange(window)
    slope, _, _, _, _ = stats.linregress(x, recent)
    return float(slope) >= slope_threshold


# ---------------------------------------------------------------------------
# Outbreak analysis
# ---------------------------------------------------------------------------

class OutbreakDetector:
    """
    Stateless outbreak detector. Can be called periodically or on-demand.
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=SARVAM_API_KEY,
            base_url=f"{SARVAM_BASE_URL}/v1",
        )

    async def analyse_district(
        self,
        district: str,
        disease: str,
        weeks: int | None = None,
    ) -> dict[str, Any]:
        """
        Full outbreak analysis for a district+disease pair.

        Returns a rich result dict with:
          - status: "outbreak" | "pre_alert" | "normal"
          - z_score: float (most recent week)
          - alert_narrative: AI-generated plain-language DHO brief
          - weekly_data: list of weekly records with z-scores appended
          - recommended_actions: list of strings
        """
        if weeks is None:
            weeks = max(52, SURVEILLANCE_LOOKBACK_DAYS // 7)

        raw_data = await hmis_bridge.get_disease_case_data(district, disease, weeks)
        if not raw_data:
            return {"status": "no_data", "district": district, "disease": disease}

        df = pd.DataFrame(raw_data)
        df["week_start"] = pd.to_datetime(df["week_start"])
        df = df.sort_values("week_start").reset_index(drop=True)

        case_series = df["suspected_cases"].astype(float)
        mean, std = compute_baseline(case_series)
        df["z_score"] = zscore_series(case_series, mean, std)

        recent_zscores = df["z_score"].iloc[-EARLY_WARNING_WEEKS:]
        latest_z = float(df["z_score"].iloc[-1])
        latest_cases = int(df["suspected_cases"].iloc[-1])
        latest_week = df["week_start"].iloc[-1].date().isoformat()

        # Classify
        if (recent_zscores >= OUTBREAK_ALERT_THRESHOLD).any():
            status = "outbreak"
        elif detect_slope_pre_alert(df["z_score"]):
            status = "pre_alert"
        else:
            status = "normal"

        # Build summary stats for Claude
        recent_df = df.tail(8)
        trend_summary = recent_df[
            ["week_start", "suspected_cases", "confirmed_cases", "deaths", "z_score"]
        ].to_dict(orient="records")

        alert_narrative = ""
        recommended_actions: list[str] = []

        if status in ("outbreak", "pre_alert"):
            alert_narrative, recommended_actions = await self._generate_alert_narrative(
                district=district,
                disease=disease,
                status=status,
                latest_week=latest_week,
                latest_cases=latest_cases,
                latest_z=latest_z,
                baseline_mean=mean,
                baseline_std=std,
                trend_summary=trend_summary,
            )

            # Send alert via HMIS bridge
            if status == "outbreak":
                await hmis_bridge.send_outbreak_alert(
                    {
                        "district": district,
                        "disease": disease,
                        "week": latest_week,
                        "suspected_cases": latest_cases,
                        "z_score": round(latest_z, 2),
                        "narrative": alert_narrative[:500],
                        "recommended_actions": recommended_actions,
                    }
                )

        return {
            "district": district,
            "disease": disease,
            "status": status,
            "latest_week": latest_week,
            "latest_suspected_cases": latest_cases,
            "z_score": round(latest_z, 2),
            "baseline_mean": round(mean, 2),
            "baseline_std": round(std, 2),
            "alert_narrative": alert_narrative,
            "recommended_actions": recommended_actions,
            "weekly_data": df.tail(12).to_dict(orient="records"),
        }

    async def _generate_alert_narrative(
        self,
        district: str,
        disease: str,
        status: str,
        latest_week: str,
        latest_cases: int,
        latest_z: float,
        baseline_mean: float,
        baseline_std: float,
        trend_summary: list[dict],
    ) -> tuple[str, list[str]]:
        """
        Use Claude to generate a DHO-ready alert brief and action list.
        """
        status_label = (
            "OUTBREAK ALERT" if status == "outbreak" else "EARLY WARNING (Pre-Alert)"
        )

        prompt = f"""You are a disease surveillance analyst for Tamil Nadu / Karnataka public health.

Generate a concise {status_label} brief for the District Health Officer (DHO).

Data:
- District: {district}
- Disease: {disease}
- Reference week: {latest_week}
- Suspected cases this week: {latest_cases}
- Z-score (deviation from baseline): {latest_z:.2f}
- Baseline mean: {baseline_mean:.1f} cases/week (±{baseline_std:.1f})
- Threshold for alert: z ≥ {OUTBREAK_ALERT_THRESHOLD}

Recent 8-week trend:
{trend_summary}

Write:
1. A 3–4 sentence plain-language alert narrative for the DHO (include case count, trend, and significance).
2. A numbered list of 4–6 specific recommended response actions tailored to {disease}.

Format your response as JSON with keys "narrative" (string) and "actions" (list of strings).
"""

        response = await self.client.chat.completions.create(
            model=SARVAM_LLM_MODEL,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = (response.choices[0].message.content or "").strip() if response.choices else ""
        try:
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            parsed = __import__("json").loads(raw)
            return parsed.get("narrative", ""), parsed.get("actions", [])
        except Exception:
            logger.warning("Could not parse Claude surveillance response; returning raw")
            return raw, []

    async def scan_all_priority_diseases(
        self,
        district: str,
        diseases: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Scan multiple priority diseases for a district and return all alerts.

        Default priority diseases per IDSP P-form / NHM surveillance.
        """
        if diseases is None:
            diseases = [
                "dengue",
                "malaria",
                "typhoid",
                "chikungunya",
                "diarrhoea",
                "acute_respiratory_illness",
                "leptospirosis",
                "scrub_typhus",
            ]

        results = []
        for disease in diseases:
            try:
                result = await self.analyse_district(district, disease)
                if result.get("status") in ("outbreak", "pre_alert"):
                    results.append(result)
            except Exception as exc:
                logger.error("Surveillance scan failed for %s/%s: %s", district, disease, exc)

        return sorted(results, key=lambda r: r.get("z_score", 0), reverse=True)


# Singleton
outbreak_detector = OutbreakDetector()
