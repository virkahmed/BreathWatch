"""
Wheeze detection model loader and inference.
Binary classification: wheeze present (1) or not present (0).
Optimized for mobile deployment with 1-second log-Mel windows.
"""
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SimpleWheezeModel(torch.nn.Module):
    """Simple CNN model for wheeze detection."""
    def __init__(self):
        super().__init__()
        # CNN architecture matching the actual model structure (16->32->64 channels, 2 outputs)
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, 2)  # 2 outputs (binary classification)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


class WheezeDetector:
    """Wheeze detector for binary wheeze detection."""
    
    def __init__(self, model_path: Optional[str] = None, model_class: Optional[torch.nn.Module] = None):
        """
        Initialize wheeze detector.
        
        Args:
            model_path: Path to PyTorch model file (.pt). If None, looks in assets folder.
            model_class: Optional PyTorch model class to instantiate. Required if loading state dict.
        """
        self.model = None
        self.model_path = model_path
        self.model_class = model_class
        self.input_shape = (1, 128, 31)  # Default shape for 1s windows
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self) -> None:
        """Load the PyTorch model."""
        if self.model_path is None:
            # Try to find model in models folder first, then assets folder
            models_path = Path(__file__).parent
            model_path = models_path / "wheeze_head.pt"
            if not model_path.exists():
                # Fallback to assets folder
                assets_path = Path(__file__).parent.parent.parent.parent / "assets"
                model_path = assets_path / "wheeze_detector.pt"
                if not model_path.exists():
                    logger.warning(
                        f"Wheeze detector model not found at {models_path / 'wheeze_head.pt'} or {assets_path / 'wheeze_detector.pt'}. "
                        "Using mock predictions. Place a trained PyTorch model at one of these locations."
                    )
                    self.model = None
                    return
            self.model_path = str(model_path)
        
        try:
            # Load PyTorch model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Check if it's a full model or just state dict
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint or 'state_dict' in checkpoint:
                    # State dict format
                    if self.model_class is None:
                        raise ValueError("model_class required when loading state dict. Provide the model architecture class.")
                    self.model = self.model_class()
                    state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
                    self.model.load_state_dict(state_dict)
                else:
                    # Assume it's a state dict directly
                    if self.model_class is None:
                        # Try to use default architecture
                        logger.info("No model_class provided, using default SimpleWheezeModel architecture")
                        self.model_class = SimpleWheezeModel
                    self.model = self.model_class()
                    # Try to load state dict, ignoring extra/missing keys
                    try:
                        self.model.load_state_dict(checkpoint, strict=False)
                        logger.info("Loaded state dict (some keys may be missing/extra)")
                    except Exception as e:
                        logger.warning(f"Could not load state dict strictly: {e}. Trying to match keys...")
                        # Try to match keys manually
                        model_dict = self.model.state_dict()
                        matched_dict = {}
                        for k, v in checkpoint.items():
                            if k in model_dict and model_dict[k].shape == v.shape:
                                matched_dict[k] = v
                        if matched_dict:
                            model_dict.update(matched_dict)
                            self.model.load_state_dict(model_dict, strict=False)
                            logger.info(f"Loaded {len(matched_dict)}/{len(checkpoint)} keys from state dict")
                        else:
                            raise ValueError(f"Could not match any keys from state dict. Available keys: {list(checkpoint.keys())[:5]}...")
            else:
                # Assume it's a full model
                self.model = checkpoint
            
            # Set to eval mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            logger.info(f"Loaded wheeze detector from {self.model_path}")
            logger.info(f"Model device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading wheeze detector model: {e}", exc_info=True)
            self.model = None
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predict wheeze probability.
        
        Args:
            features: Feature array (1s log-Mel spectrogram)
        
        Returns:
            Probability of wheeze being present [0, 1]
        """
        if self.model is None:
            # Mock prediction for development
            logger.warning("Using mock wheeze prediction (model not loaded)")
            return self._mock_predict(features)
        
        try:
            # Prepare input
            input_tensor = self._prepare_input(features)
            
            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Convert to numpy and extract probability
            if isinstance(output, torch.Tensor):
                output_array = output.cpu().numpy()
            else:
                output_array = np.array(output)
            
            # Handle 2-output binary classification
            # Output shape: (batch, 2) - [no_wheeze_prob, wheeze_prob]
            output_array = output_array.flatten()
            if len(output_array) == 2:
                # Take the second value (wheeze present probability)
                probability = float(output_array[1])
            else:
                # Fallback to first value if unexpected shape
                probability = float(output_array[0])
            
            # Ensure probability is in [0, 1] range
            probability = max(0.0, min(1.0, probability))
            
            # Ensure valid range
            probability = max(0.0, min(1.0, probability))
            
            logger.debug(f"Wheeze prediction: {probability:.3f}")
            return probability
            
        except Exception as e:
            logger.error(f"Error in wheeze prediction: {e}", exc_info=True)
            return self._mock_predict(features)
    
    def _prepare_input(self, features: np.ndarray) -> torch.Tensor:
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
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(features.astype(np.float32))
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def _mock_predict(self, features: np.ndarray) -> float:
        """Mock prediction for development/testing."""
        # Simple heuristic: high frequency energy might indicate wheeze
        if len(features.shape) >= 2:
            high_freq_energy = np.mean(features[-features.shape[0]//4:, :])
            prob = min(0.7, max(0.1, high_freq_energy))
        else:
            prob = 0.3
        
        return prob
