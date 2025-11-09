"""
Cough classification model loader and inference.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import onnxruntime as ort

logger = logging.getLogger(__name__)


class CoughClassifier:
    """Cough classifier for healthy vs sick cough detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize cough classifier.
        
        Args:
            model_path: Path to ONNX model file. If None, looks in assets folder.
        """
        self.model = None
        self.model_path = model_path
        self.input_shape = (1, 128, 87)  # Default shape (channels, height, width)
        self.load_model()
    
    def load_model(self) -> None:
        """Load the ONNX model."""
        if self.model_path is None:
            # Try to find model in assets folder
            assets_path = Path(__file__).parent.parent.parent.parent / "assets"
            model_path = assets_path / "cough_classifier.onnx"
            if not model_path.exists():
                logger.warning(
                    f"Cough model not found at {model_path}. "
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
            
            logger.info(f"Loaded cough classifier from {self.model_path}")
            logger.info(f"Model input shape: {self.input_shape}")
        except Exception as e:
            logger.error(f"Error loading cough model: {e}")
            self.model = None
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict cough class probabilities.
        
        Args:
            features: Feature array (mel spectrogram or MFCC)
        
        Returns:
            Tuple of (healthy_probability, sick_probability)
        """
        if self.model is None:
            # Mock prediction for development
            logger.warning("Using mock cough prediction (model not loaded)")
            return self._mock_predict(features)
        
        try:
            # Prepare input
            input_data = self._prepare_input(features)
            
            # Run inference
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_data})
            
            # Get probabilities
            probabilities = output[0][0]  # Assuming single output with 2 classes
            
            # Normalize to probabilities
            if len(probabilities) == 2:
                healthy_prob = float(probabilities[0])
                sick_prob = float(probabilities[1])
            else:
                # Single output, assume sigmoid
                healthy_prob = 1.0 - float(probabilities[0])
                sick_prob = float(probabilities[0])
            
            # Ensure probabilities sum to 1
            total = healthy_prob + sick_prob
            if total > 0:
                healthy_prob /= total
                sick_prob /= total
            
            logger.debug(f"Cough prediction: healthy={healthy_prob:.3f}, sick={sick_prob:.3f}")
            return healthy_prob, sick_prob
            
        except Exception as e:
            logger.error(f"Error in cough prediction: {e}")
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
    
    def _mock_predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Mock prediction for development/testing."""
        # Simple heuristic: if audio has high energy in mid frequencies, might be sick
        if len(features.shape) >= 2:
            mid_freq_energy = np.mean(features[:, features.shape[1]//4:3*features.shape[1]//4])
            sick_prob = min(0.7, max(0.3, mid_freq_energy))
        else:
            sick_prob = 0.5
        
        healthy_prob = 1.0 - sick_prob
        return healthy_prob, sick_prob

