"""
Respiratory anomaly classification model loader and inference.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import onnxruntime as ort

logger = logging.getLogger(__name__)


class RespiratoryClassifier:
    """Respiratory anomaly classifier for normal, wheeze, crackle detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize respiratory classifier.
        
        Args:
            model_path: Path to ONNX model file. If None, looks in assets folder.
        """
        self.model = None
        self.model_path = model_path
        self.input_shape = (1, 128, 87)  # Default shape
        self.class_names = ["normal", "wheeze", "crackle", "crackle_wheeze"]
        self.load_model()
    
    def load_model(self) -> None:
        """Load the ONNX model."""
        if self.model_path is None:
            # Try to find model in assets folder
            assets_path = Path(__file__).parent.parent.parent.parent / "assets"
            model_path = assets_path / "respiratory_classifier.onnx"
            if not model_path.exists():
                logger.warning(
                    f"Respiratory model not found at {model_path}. "
                    "Using mock predictions. Place a trained ONNX model at this location."
                )
                self.model = None
                return
            self.model_path = str(model_path)
        
        try:
            # Create ONNX Runtime session
            self.model = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input shape from model
            input_meta = self.model.get_inputs()[0]
            if input_meta.shape:
                self.input_shape = tuple(input_meta.shape[1:])  # Remove batch dimension
            
            logger.info(f"Loaded respiratory classifier from {self.model_path}")
            logger.info(f"Model input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Error loading respiratory model: {e}")
            self.model = None
    
    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Predict respiratory anomaly class probabilities.
        
        Args:
            features: Feature array (mel spectrogram or MFCC)
        
        Returns:
            Dictionary with class probabilities
        """
        if self.model is None:
            # Mock prediction for development
            logger.warning("Using mock respiratory prediction (model not loaded)")
            return self._mock_predict(features)
        
        try:
            # Prepare input
            input_data = self._prepare_input(features)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_data})
            
            # Get probabilities
            probabilities = output[0][0]  # Assuming single output with 4 classes
            
            # Map to class names
            if len(probabilities) == 4:
                probs = {
                    "normal": float(probabilities[0]),
                    "wheeze": float(probabilities[1]),
                    "crackle": float(probabilities[2]),
                    "crackle_wheeze": float(probabilities[3])
                }
            else:
                # Fallback: assume binary or different structure
                probs = {"normal": 0.5, "wheeze": 0.3, "crackle": 0.1, "crackle_wheeze": 0.1}
            
            # Normalize probabilities
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}
            
            logger.debug(f"Respiratory prediction: {probs}")
            return probs
            
        except Exception as e:
            logger.error(f"Error in respiratory prediction: {e}")
            return self._mock_predict(features)
    
    def _prepare_input(self, features: np.ndarray) -> np.ndarray:
        """Prepare features for model input."""
        # Ensure correct shape
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        # Resize if needed
        if features.shape[1:] != self.input_shape:
            from scipy.ndimage import zoom
            current_shape = features.shape[1:]
            zoom_factors = (
                self.input_shape[0] / current_shape[0],
                self.input_shape[1] / current_shape[1]
            )
            features = zoom(features, (1,) + zoom_factors, order=1)
        
        # Add batch dimension if missing
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=0)
        
        # Ensure correct dtype
        features = features.astype(np.float32)
        
        return features
    
    def _mock_predict(self, features: np.ndarray) -> Dict[str, float]:
        """Mock prediction for development/testing."""
        # Simple heuristic based on spectral characteristics
        if len(features.shape) >= 2:
            # High frequency energy might indicate wheeze
            high_freq_energy = np.mean(features[-features.shape[0]//4:, :])
            # Mid frequency variation might indicate crackle
            mid_freq_var = np.var(features[features.shape[0]//4:3*features.shape[0]//4, :])
            
            wheeze_prob = min(0.6, max(0.1, high_freq_energy))
            crackle_prob = min(0.4, max(0.05, mid_freq_var * 10))
            crackle_wheeze_prob = min(0.3, wheeze_prob * crackle_prob)
            normal_prob = 1.0 - (wheeze_prob + crackle_prob + crackle_wheeze_prob)
            normal_prob = max(0.1, normal_prob)
        else:
            normal_prob = 0.5
            wheeze_prob = 0.3
            crackle_prob = 0.1
            crackle_wheeze_prob = 0.1
        
        # Normalize
        total = normal_prob + wheeze_prob + crackle_prob + crackle_wheeze_prob
        return {
            "normal": normal_prob / total,
            "wheeze": wheeze_prob / total,
            "crackle": crackle_prob / total,
            "crackle_wheeze": crackle_wheeze_prob / total
        }

