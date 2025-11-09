"""
Dedalus AI client for health interpretation using the Dedalus Labs SDK.
"""
import logging
import os
from typing import Optional, List
from app.schemas import DedalusInterpretation

try:
    from dedalus_labs import Dedalus, DedalusRunner
    DEDALUS_AVAILABLE = True
except ImportError:
    DEDALUS_AVAILABLE = False
    Dedalus = None
    DedalusRunner = None

logger = logging.getLogger(__name__)


def analyze_respiratory_data(
    cough_count: int,
    wheeze_count: int,
    wheeze_probability: float,
    attribute_wet_percent: float,
    attribute_stridor_percent: float,
    attribute_choking_percent: float,
    attribute_congestion_percent: float,
    attribute_wheezing_selfreport_percent: float,
    sleep_duration_minutes: Optional[float] = None,
    patient_age: Optional[int] = None,
    patient_sex: Optional[str] = None
) -> dict:
    """
    Analyze respiratory monitoring data and return interpretation.
    
    This function is provided as a tool to the Dedalus AI runner.
    The LLM can call this function to analyze respiratory data.
    
    Args:
        cough_count: Total number of coughs detected
        wheeze_count: Total number of wheeze events detected
        wheeze_probability: Average wheeze probability (0.0 to 1.0)
        attribute_wet_percent: Percentage of windows/events with wet cough attribute
        attribute_stridor_percent: Percentage of windows/events with stridor attribute
        attribute_choking_percent: Percentage of windows/events with choking attribute
        attribute_congestion_percent: Percentage of windows/events with congestion attribute
        attribute_wheezing_selfreport_percent: Percentage of windows/events with self-reported wheezing attribute
        sleep_duration_minutes: Sleep duration in minutes
        patient_age: Patient age in years
        patient_sex: Patient sex (M/F)
    
    Returns:
        Dictionary with interpretation, severity, and recommendations
    """
    severity = "normal"
    interpretation_parts = []
    recommendations = []
    
    # Analyze cough patterns
    if cough_count > 20:
        interpretation_parts.append(f"Detected frequent coughing ({cough_count} cough events)")
        recommendations.append("Consider consulting a healthcare provider if persistent")
        severity = "moderate"
    elif cough_count > 10:
        interpretation_parts.append(f"Detected moderate coughing ({cough_count} cough events)")
        severity = "mild"
    
    # Analyze wheeze
    if wheeze_count > 0 or wheeze_probability > 0.7:
        severity = "moderate" if severity == "normal" else "severe"
        interpretation_parts.append("significant wheeze patterns detected")
        recommendations.append("Wheezing may indicate airway inflammation or obstruction")
    elif wheeze_probability > 0.4:
        interpretation_parts.append("mild wheeze patterns detected")
        if severity == "normal":
            severity = "mild"
    
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
    
    return {
        "interpretation": interpretation,
        "severity": severity,
        "recommendations": recommendations
    }


class DedalusClient:
    """Client for interacting with Dedalus AI using the SDK."""
    
    def __init__(self, api_key: Optional[str] = None, openai_api_key: Optional[str] = None, model: str = "openai/gpt-4o"):
        """
        Initialize Dedalus client.
        
        Args:
            api_key: API key for Dedalus AI (can be set via environment variable DEDALUS_API_KEY)
            openai_api_key: OpenAI API key to use when Dedalus calls OpenAI (can be set via environment variable OPENAI_API_KEY)
            model: Model to use for interpretation (default: gpt-4o)
        """
        # Initialize attributes first to ensure they always exist
        self.client = None
        self.runner = None
        self.api_key = None
        self.model = model
        
        if not DEDALUS_AVAILABLE:
            logger.warning("dedalus-labs package not installed. Install with: pip install dedalus-labs")
            return
        
        self.api_key = api_key or self._get_api_key_from_env()
        
        # Get OpenAI API key from parameter or environment
        openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            # Set OpenAI API key in environment so Dedalus can use it
            os.environ["OPENAI_API_KEY"] = openai_key
            logger.info("OpenAI API key configured for Dedalus")
        else:
            logger.warning("OpenAI API key not provided. Dedalus may use its own key or fail if required.")
        
        if self.api_key:
            try:
                # Set API key in environment if not already set
                if not os.getenv("DEDALUS_API_KEY"):
                    os.environ["DEDALUS_API_KEY"] = self.api_key
                
                self.client = Dedalus()
                # Note: DedalusRunner is used for tool-calling workflows
                # Store both client and runner for compatibility
                try:
                    self.runner = DedalusRunner(self.client)
                except Exception as runner_error:
                    logger.warning(f"Could not create DedalusRunner: {runner_error}. Using client directly.")
                    self.runner = None
                logger.info("Dedalus AI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Dedalus client: {e}", exc_info=True)
                self.client = None
                self.runner = None
        else:
            logger.warning("Dedalus API key not provided")
    
    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variable."""
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
        Send analysis results to Dedalus AI for interpretation using the SDK.
        
        Accepts summary data from mobile app (which processes audio locally).
        This method uses Dedalus AI's tool-calling framework to interpret results.
        
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
            DedalusInterpretation object or None if interpretation fails
        """
        # If runner is not available, use tool function directly as fallback
        if not self.runner:
            if not self.api_key:
                logger.warning("Dedalus API key not provided, using tool function directly")
            elif not DEDALUS_AVAILABLE:
                logger.warning("dedalus-labs package not available, using tool function directly")
            else:
                logger.warning("Dedalus runner not initialized, using tool function directly")
            
            # Fallback: use tool function directly
            analysis_result = analyze_respiratory_data(
                cough_count=cough_count,
                wheeze_count=wheeze_count,
                wheeze_probability=wheeze_probability,
                attribute_wet_percent=attribute_wet_percent,
                attribute_stridor_percent=attribute_stridor_percent,
                attribute_choking_percent=attribute_choking_percent,
                attribute_congestion_percent=attribute_congestion_percent,
                attribute_wheezing_selfreport_percent=attribute_wheezing_selfreport_percent,
                sleep_duration_minutes=sleep_duration_minutes,
                patient_age=patient_age,
                patient_sex=patient_sex
            )
            
            return DedalusInterpretation(
                interpretation=analysis_result["interpretation"],
                severity=analysis_result["severity"],
                recommendations=analysis_result["recommendations"]
            )
        
        try:
            # Format input prompt for Dedalus AI
            patient_info = ""
            if patient_age:
                patient_info += f"Patient age: {patient_age} years. "
            if patient_sex:
                patient_info += f"Patient sex: {patient_sex}. "
            if sleep_duration_minutes:
                hours = sleep_duration_minutes / 60.0
                patient_info += f"Sleep duration: {hours:.1f} hours ({sleep_duration_minutes:.0f} minutes). "
            
            input_prompt = f"""Analyze this respiratory monitoring data from overnight sleep monitoring:

{patient_info}
- Cough count: {cough_count} coughs detected
- Wheeze detection: {wheeze_count} windows with wheeze ({wheeze_probability*100:.1f}% of time)
- Normal breathing: {normal_probability*100:.1f}% of time

Cough attributes (percentage of windows/events flagged):
- Wet cough: {attribute_wet_percent:.1f}%
- Stridor: {attribute_stridor_percent:.1f}%
- Choking: {attribute_choking_percent:.1f}%
- Congestion: {attribute_congestion_percent:.1f}%
- Self-reported wheezing: {attribute_wheezing_selfreport_percent:.1f}%

Please analyze this data using the analyze_respiratory_data tool and provide a health interpretation including severity assessment and recommendations."""
            
            # Run Dedalus AI with the analysis tool
            result = self.runner.run(
                input=input_prompt,
                model=self.model,
                tools=[analyze_respiratory_data]
            )
            
            # Extract interpretation from result
            # The runner returns a result object with final_output attribute
            if hasattr(result, 'final_output'):
                output = result.final_output
            else:
                output = str(result)
            
            # The tool function will be called by the LLM, so we can also call it directly
            # to get structured results as a fallback
            analysis_result = analyze_respiratory_data(
                cough_count=cough_count,
                wheeze_count=wheeze_count,
                wheeze_probability=wheeze_probability,
                attribute_wet_percent=attribute_wet_percent,
                attribute_stridor_percent=attribute_stridor_percent,
                attribute_choking_percent=attribute_choking_percent,
                attribute_congestion_percent=attribute_congestion_percent,
                attribute_wheezing_selfreport_percent=attribute_wheezing_selfreport_percent,
                sleep_duration_minutes=sleep_duration_minutes,
                patient_age=patient_age,
                patient_sex=patient_sex
            )
            
            # Use Dedalus AI's interpretation if it contains structured information
            # Otherwise use the tool result directly
            if isinstance(output, dict) and "interpretation" in output:
                # Tool was called and returned dict directly
                interpretation = DedalusInterpretation(
                    interpretation=output.get("interpretation", analysis_result["interpretation"]),
                    severity=output.get("severity", analysis_result["severity"]),
                    recommendations=output.get("recommendations", analysis_result["recommendations"])
                )
            elif "interpretation" in str(output).lower() or "recommendation" in str(output).lower():
                # LLM provided text interpretation, use it with tool's severity/recommendations
                interpretation = DedalusInterpretation(
                    interpretation=str(output),
                    severity=analysis_result.get("severity"),
                    recommendations=analysis_result.get("recommendations", [])
                )
            else:
                # Use tool result directly
                interpretation = DedalusInterpretation(
                    interpretation=analysis_result["interpretation"],
                    severity=analysis_result["severity"],
                    recommendations=analysis_result["recommendations"]
                )
            
            logger.info("Successfully received interpretation from Dedalus AI")
            return interpretation
            
        except Exception as e:
            error_str = str(e)
            # Check if it's a balance/account error (402, 401, etc.)
            if "402" in error_str or "balance" in error_str.lower() or "401" in error_str or "unauthorized" in error_str.lower():
                logger.warning(f"Dedalus AI API error (likely account/balance issue): {e}")
                logger.info("Falling back to tool function for interpretation")
            else:
                logger.error(f"Error calling Dedalus AI: {e}", exc_info=True)
            
            # Fallback: use tool function directly when API fails
            analysis_result = analyze_respiratory_data(
                cough_count=cough_count,
                wheeze_count=wheeze_count,
                wheeze_probability=wheeze_probability,
                attribute_wet_percent=attribute_wet_percent,
                attribute_stridor_percent=attribute_stridor_percent,
                attribute_choking_percent=attribute_choking_percent,
                attribute_congestion_percent=attribute_congestion_percent,
                attribute_wheezing_selfreport_percent=attribute_wheezing_selfreport_percent,
                sleep_duration_minutes=sleep_duration_minutes,
                patient_age=patient_age,
                patient_sex=patient_sex
            )
            
            return DedalusInterpretation(
                interpretation=analysis_result["interpretation"],
                severity=analysis_result["severity"],
                recommendations=analysis_result["recommendations"]
            )
