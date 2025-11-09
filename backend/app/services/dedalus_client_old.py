"""
Dedalus AI client for health interpretation.
"""
import logging
import httpx
from typing import Dict, Optional, List
from app.schemas import DedalusInterpretation

logger = logging.getLogger(__name__)


class DedalusClient:
    """Client for interacting with Dedalus AI API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.dedalus.ai"):
        """
        Initialize Dedalus client.
        
        Args:
            api_key: API key for Dedalus AI (can be set via environment variable DEDALUS_API_KEY)
            base_url: Base URL for Dedalus API
        """
        self.api_key = api_key or self._get_api_key_from_env()
        self.base_url = base_url.rstrip('/')
        self.timeout = 30.0
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
        import os
        return os.getenv("DEDALUS_API_KEY")
    
    def interpret_results(
        self,
        cough_count: int,
        wheeze_count: int = 0,
        wheeze_probability: float = 0.0,
        cough_healthy_count: int = 0,
        cough_sick_count: int = 0,
        crackle_probability: float = 0.0,
        normal_probability: float = 0.0,
        sleep_duration_minutes: Optional[float] = None,
        patient_age: Optional[int] = None,
        patient_sex: Optional[str] = None,
        attribute_wet_percent: float = 0.0,
        attribute_stridor_percent: float = 0.0,
        attribute_choking_percent: float = 0.0,
        attribute_congestion_percent: float = 0.0,
        attribute_wheezing_selfreport_percent: float = 0.0
    ) -> Optional[DedalusInterpretation]:
        """
        Send analysis results to Dedalus AI for interpretation.
        
        Accepts summary data from mobile app (which processes audio locally).
        This method only interprets the results, not processes audio.
        
        Args:
            cough_count: Total number of coughs detected
            wheeze_count: Total number of wheeze events detected
            wheeze_probability: Average wheeze probability
            cough_healthy_count: Number of healthy coughs (legacy, not used in binary VAD)
            cough_sick_count: Number of sick coughs (legacy, not used in binary VAD)
            crackle_probability: Overall crackle probability (legacy, not detected in binary model)
            normal_probability: Overall normal breathing probability
            sleep_duration_minutes: Sleep duration
            patient_age: Patient age
            patient_sex: Patient sex
            attribute_wet_percent: Percentage of windows/events with wet cough attribute
            attribute_stridor_percent: Percentage of windows/events with stridor attribute
            attribute_choking_percent: Percentage of windows/events with choking attribute
            attribute_congestion_percent: Percentage of windows/events with congestion attribute
            attribute_wheezing_selfreport_percent: Percentage of windows/events with self-reported wheezing attribute
        
        Returns:
            DedalusInterpretation object or None if API call fails
        """
        if not self.api_key:
            logger.error("Dedalus API key not provided")
            return None
        
        try:
            # Prepare request data
            request_data = {
                "cough_count": cough_count,
                "cough_healthy_count": cough_healthy_count,
                "cough_sick_count": cough_sick_count,
                "wheeze_count": wheeze_count,
                "wheeze_probability": wheeze_probability,
                "crackle_probability": crackle_probability,
                "normal_probability": normal_probability,
                "sleep_duration_minutes": sleep_duration_minutes,
                "patient_age": patient_age,
                "patient_sex": patient_sex,
                "attribute_wet_percent": attribute_wet_percent,
                "attribute_stridor_percent": attribute_stridor_percent,
                "attribute_choking_percent": attribute_choking_percent,
                "attribute_congestion_percent": attribute_congestion_percent,
                "attribute_wheezing_selfreport_percent": attribute_wheezing_selfreport_percent
            }
            
            # Make API call
            response = httpx.post(
                f"{self.base_url}/v1/interpret",
                json=request_data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Parse response
            interpretation = DedalusInterpretation(
                interpretation=result.get("interpretation", "No interpretation available"),
                severity=result.get("severity"),
                recommendations=result.get("recommendations", [])
            )
            
            logger.info("Successfully received interpretation from Dedalus AI")
            return interpretation
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Dedalus API: {e}")
            return None
        except Exception as e:
            logger.error(f"Error calling Dedalus API: {e}")
            return None
    
    def _mock_interpretation(
        self,
        cough_count: int,
        wheeze_count: int = 0,
        wheeze_prob: float = 0.0,
        attribute_wet_percent: float = 0.0,
        attribute_stridor_percent: float = 0.0,
        attribute_choking_percent: float = 0.0,
        attribute_congestion_percent: float = 0.0,
        attribute_wheezing_selfreport_percent: float = 0.0
    ) -> DedalusInterpretation:
        """Generate mock interpretation for development/testing."""
        severity = "normal"
        interpretation_parts = []
        recommendations = []
        
        # Analyze cough patterns
        if cough_count > 20:
            interpretation_parts.append(f"Detected frequent coughing ({cough_count} cough events)")
            recommendations.append("Consider consulting a healthcare provider if persistent")
        elif cough_count > 10:
            interpretation_parts.append(f"Detected moderate coughing ({cough_count} cough events)")
        
        # Analyze wheeze
        if wheeze_count > 0 or wheeze_prob > 0.7:
            severity = "moderate" if severity == "normal" else "severe"
            interpretation_parts.append("significant wheeze patterns detected")
            recommendations.append("Wheezing may indicate airway inflammation or obstruction")
        elif wheeze_prob > 0.4:
            interpretation_parts.append("mild wheeze patterns detected")
        
        # Analyze cough attributes
        attribute_notes = []
        if attribute_wet_percent > 30:
            attribute_notes.append(f"wet cough characteristics ({attribute_wet_percent:.1f}%)")
            if severity == "normal":
                severity = "mild"
            recommendations.append("Wet cough may indicate productive respiratory condition")
        
        if attribute_stridor_percent > 20:
            attribute_notes.append(f"stridor-like sounds ({attribute_stridor_percent:.1f}%)")
            if severity == "normal":
                severity = "moderate"
            recommendations.append("Stridor may indicate upper airway obstruction - seek medical attention")
        
        if attribute_choking_percent > 15:
            attribute_notes.append(f"choking episodes ({attribute_choking_percent:.1f}%)")
            severity = "moderate" if severity == "normal" else "severe"
            recommendations.append("Choking episodes require immediate medical evaluation")
        
        if attribute_congestion_percent > 25:
            attribute_notes.append(f"congestion patterns ({attribute_congestion_percent:.1f}%)")
            if not any("congestion" in r.lower() for r in recommendations):
                recommendations.append("Nasal congestion may be contributing to respiratory symptoms")
        
        if attribute_wheezing_selfreport_percent > 20:
            attribute_notes.append(f"self-reported wheezing patterns ({attribute_wheezing_selfreport_percent:.1f}%)")
        
        if attribute_notes:
            interpretation_parts.append("Cough attributes: " + ", ".join(attribute_notes))
        
        # Generate interpretation text
        if interpretation_parts:
            interpretation = " — ".join(interpretation_parts) + "."
            if not recommendations:
                recommendations.append("Monitor symptoms and consult a healthcare provider if they persist or worsen")
        else:
            interpretation = "No significant respiratory anomalies detected. Breathing patterns appear normal."
            recommendations.append("Continue monitoring")
        
        return DedalusInterpretation(
            interpretation=interpretation,
            severity=severity,
            recommendations=recommendations
        )
