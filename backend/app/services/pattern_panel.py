"""
Pattern panel service that fuses audio metrics with symptom form.
Generates pattern scores (asthma-like, COPD-like, etc.) with explanations.
"""
import logging
from typing import List, Dict
from app.schemas import PatternScore, SymptomForm, AttributePrevalence

logger = logging.getLogger(__name__)


def calculate_pattern_scores(
    coughs_per_hour: float,
    wheeze_time_percent: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> List[PatternScore]:
    """
    Calculate pattern scores by fusing audio metrics and symptoms.
    
    Args:
        coughs_per_hour: Cough frequency
        wheeze_time_percent: Percentage of time with wheeze
        attribute_prevalence: Attribute prevalence statistics
        symptom_form: User symptom inputs
    
    Returns:
        List of pattern scores with uncertainty and explanations
    """
    patterns = []
    
    # Asthma-like pattern
    asthma_score, asthma_uncertainty = _calculate_asthma_score(
        coughs_per_hour, wheeze_time_percent, attribute_prevalence, symptom_form
    )
    patterns.append(PatternScore(
        pattern_name="Asthma-like",
        score=asthma_score,
        uncertainty_lower=max(0.0, asthma_score - asthma_uncertainty),
        uncertainty_upper=min(1.0, asthma_score + asthma_uncertainty),
        why=_explain_asthma(coughs_per_hour, wheeze_time_percent, symptom_form)
    ))
    
    # COPD-like pattern
    copd_score, copd_uncertainty = _calculate_copd_score(
        coughs_per_hour, wheeze_time_percent, attribute_prevalence, symptom_form
    )
    patterns.append(PatternScore(
        pattern_name="COPD-like",
        score=copd_score,
        uncertainty_lower=max(0.0, copd_score - copd_uncertainty),
        uncertainty_upper=min(1.0, copd_score + copd_uncertainty),
        why=_explain_copd(coughs_per_hour, wheeze_time_percent, symptom_form)
    ))
    
    # URTI-like pattern
    urti_score, urti_uncertainty = _calculate_urti_score(
        coughs_per_hour, attribute_prevalence, symptom_form
    )
    patterns.append(PatternScore(
        pattern_name="URTI-like",
        score=urti_score,
        uncertainty_lower=max(0.0, urti_score - urti_uncertainty),
        uncertainty_upper=min(1.0, urti_score + urti_uncertainty),
        why=_explain_urti(coughs_per_hour, attribute_prevalence, symptom_form)
    ))
    
    # COVID-like pattern (research-only)
    covid_score, covid_uncertainty = _calculate_covid_score(
        coughs_per_hour, attribute_prevalence, symptom_form
    )
    patterns.append(PatternScore(
        pattern_name="COVID-like",
        score=covid_score,
        uncertainty_lower=max(0.0, covid_score - covid_uncertainty),
        uncertainty_upper=min(1.0, covid_score + covid_uncertainty),
        why=_explain_covid(coughs_per_hour, attribute_prevalence, symptom_form)
    ))
    
    # Pneumonia-like pattern
    pneumonia_score, pneumonia_uncertainty = _calculate_pneumonia_score(
        coughs_per_hour, attribute_prevalence, symptom_form
    )
    patterns.append(PatternScore(
        pattern_name="Pneumonia-like",
        score=pneumonia_score,
        uncertainty_lower=max(0.0, pneumonia_score - pneumonia_uncertainty),
        uncertainty_upper=min(1.0, pneumonia_score + pneumonia_uncertainty),
        why=_explain_pneumonia(coughs_per_hour, attribute_prevalence, symptom_form)
    ))
    
    # Unclear pattern (default if nothing matches well)
    unclear_score = 1.0 - max(p.score for p in patterns)
    patterns.append(PatternScore(
        pattern_name="Unclear",
        score=unclear_score,
        uncertainty_lower=max(0.0, unclear_score - 0.2),
        uncertainty_upper=min(1.0, unclear_score + 0.2),
        why="Pattern does not clearly match any known respiratory condition"
    ))
    
    return patterns


def _calculate_asthma_score(
    coughs_per_hour: float,
    wheeze_time_percent: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> tuple[float, float]:
    """Calculate asthma-like pattern score."""
    score = 0.0
    
    # Wheeze is strong indicator
    if wheeze_time_percent > 20:
        score += 0.4
    elif wheeze_time_percent > 10:
        score += 0.2
    
    # Nocturnal worsening
    if symptom_form.nocturnal_worsening:
        score += 0.2
    
    # Asthma history
    if symptom_form.asthma_history:
        score += 0.2
    
    # Chest tightness
    if symptom_form.chest_tightness:
        score += 0.1
    
    # Cough frequency (moderate)
    if 5 <= coughs_per_hour <= 30:
        score += 0.1
    
    # Uncertainty based on data quality
    uncertainty = 0.15 if wheeze_time_percent > 0 else 0.25
    
    return min(1.0, score), uncertainty


def _calculate_copd_score(
    coughs_per_hour: float,
    wheeze_time_percent: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> tuple[float, float]:
    """Calculate COPD-like pattern score."""
    score = 0.0
    
    # COPD history
    if symptom_form.copd_history:
        score += 0.3
    
    # Smoking
    if symptom_form.smoker:
        score += 0.2
    
    # Chronic cough
    if coughs_per_hour > 20:
        score += 0.2
    
    # Wheeze
    if wheeze_time_percent > 15:
        score += 0.15
    
    # Age (older)
    if symptom_form.age_band in ["51+", "31-50"]:
        score += 0.1
    
    # Duration (chronic)
    if symptom_form.duration > 7:
        score += 0.05
    
    uncertainty = 0.2
    
    return min(1.0, score), uncertainty


def _calculate_urti_score(
    coughs_per_hour: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> tuple[float, float]:
    """Calculate URTI-like pattern score."""
    score = 0.0
    
    # Sore throat
    if symptom_form.sore_throat:
        score += 0.3
    
    # Congestion
    if attribute_prevalence.congestion > 30:
        score += 0.2
    
    # Fever
    if symptom_form.fever:
        score += 0.2
    
    # Acute duration
    if 1 <= symptom_form.duration <= 7:
        score += 0.15
    
    # Cough frequency
    if 10 <= coughs_per_hour <= 40:
        score += 0.15
    
    uncertainty = 0.2
    
    return min(1.0, score), uncertainty


def _calculate_covid_score(
    coughs_per_hour: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> tuple[float, float]:
    """Calculate COVID-like pattern score (research-only)."""
    score = 0.0
    
    # Dry cough (low wet attribute)
    if attribute_prevalence.wet < 20:
        score += 0.2
    
    # Fever
    if symptom_form.fever:
        score += 0.2
    
    # Chest tightness
    if symptom_form.chest_tightness:
        score += 0.15
    
    # Moderate cough frequency
    if 5 <= coughs_per_hour <= 25:
        score += 0.15
    
    # Acute onset
    if symptom_form.duration <= 14:
        score += 0.1
    
    # Congestion (less common in COVID)
    if attribute_prevalence.congestion < 20:
        score += 0.1
    
    # High uncertainty (research pattern)
    uncertainty = 0.3
    
    return min(1.0, score), uncertainty


def _calculate_pneumonia_score(
    coughs_per_hour: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> tuple[float, float]:
    """Calculate pneumonia-like pattern score."""
    score = 0.0
    
    # Wet cough
    if attribute_prevalence.wet > 40:
        score += 0.3
    
    # Fever
    if symptom_form.fever:
        score += 0.2
    
    # Chest tightness
    if symptom_form.chest_tightness:
        score += 0.15
    
    # High cough frequency
    if coughs_per_hour > 30:
        score += 0.15
    
    # Acute
    if symptom_form.duration <= 14:
        score += 0.1
    
    # Choking/congestion
    if attribute_prevalence.choking > 20 or attribute_prevalence.congestion > 30:
        score += 0.1
    
    uncertainty = 0.25
    
    return min(1.0, score), uncertainty


def _explain_asthma(
    coughs_per_hour: float,
    wheeze_time_percent: float,
    symptom_form: SymptomForm
) -> str:
    """Generate explanation for asthma pattern."""
    reasons = []
    if wheeze_time_percent > 10:
        reasons.append(f"↑ wheeze-time ({wheeze_time_percent:.1f}%)")
    if symptom_form.nocturnal_worsening:
        reasons.append("nocturnal chest tightness")
    if symptom_form.asthma_history:
        reasons.append("asthma history")
    return " + ".join(reasons) if reasons else "Moderate wheeze pattern detected"


def _explain_copd(
    coughs_per_hour: float,
    wheeze_time_percent: float,
    symptom_form: SymptomForm
) -> str:
    """Generate explanation for COPD pattern."""
    reasons = []
    if symptom_form.copd_history:
        reasons.append("COPD history")
    if symptom_form.smoker:
        reasons.append("smoking history")
    if coughs_per_hour > 20:
        reasons.append(f"chronic cough ({coughs_per_hour:.1f}/hr)")
    return " + ".join(reasons) if reasons else "Chronic respiratory pattern"


def _explain_urti(
    coughs_per_hour: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> str:
    """Generate explanation for URTI pattern."""
    reasons = []
    if symptom_form.sore_throat:
        reasons.append("sore throat")
    if attribute_prevalence.congestion > 30:
        reasons.append("congestion")
    if symptom_form.fever:
        reasons.append("fever")
    return " + ".join(reasons) if reasons else "Upper respiratory symptoms"


def _explain_covid(
    coughs_per_hour: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> str:
    """Generate explanation for COVID pattern (research-only)."""
    reasons = []
    if attribute_prevalence.wet < 20:
        reasons.append("dry cough")
    if symptom_form.fever:
        reasons.append("fever")
    if symptom_form.chest_tightness:
        reasons.append("chest tightness")
    return " + ".join(reasons) if reasons else "Respiratory symptoms (research pattern)"


def _explain_pneumonia(
    coughs_per_hour: float,
    attribute_prevalence: AttributePrevalence,
    symptom_form: SymptomForm
) -> str:
    """Generate explanation for pneumonia pattern."""
    reasons = []
    if attribute_prevalence.wet > 40:
        reasons.append("wet cough")
    if symptom_form.fever:
        reasons.append("fever")
    if symptom_form.chest_tightness:
        reasons.append("chest tightness")
    return " + ".join(reasons) if reasons else "Lower respiratory symptoms"

