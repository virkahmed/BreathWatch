"""
Cough model loader and inference with attribute probabilities.
Outputs: p_cough + 5 attribute probabilities (wet, stridor, choking, congestion, wheezing_selfreport).
Optimized for mobile deployment with 1-second log-Mel windows.
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class SimpleCoughModel(nn.Module):
    """Simple CNN model for cough detection with attributes."""
    def __init__(self, num_outputs=6):
        super().__init__()
        # CNN architecture matching the actual model structure (16->32->64 channels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_outputs)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)


class CoughVAD:
    """Cough model with attribute probabilities."""
    
    def __init__(self, model_path: Optional[str] = None, model_class: Optional[torch.nn.Module] = None):
        """
        Initialize cough model.
        
        Args:
            model_path: Path to PyTorch model file (.pt). If None, looks in assets folder.
            model_class: Optional PyTorch model class to instantiate. Required if loading state dict.
        """
        self.model = None
        self.model_path = model_path
        self.model_class = model_class
        self.input_shape = (1, 128, 31)  # Default shape for 1s windows
        self.num_outputs = 6  # p_cough + 5 attributes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self) -> None:
        """Load the PyTorch model."""
        if self.model_path is None:
            # Try to find model in models folder first, then assets folder
            models_path = Path(__file__).parent
            model_path = models_path / "cough_multitask.pt"
            if not model_path.exists():
                # Fallback to assets folder
                assets_path = Path(__file__).parent.parent.parent.parent / "assets"
                model_path = assets_path / "cough_model.pt"
                if not model_path.exists():
                    logger.warning(
                        f"Cough model not found at {models_path / 'cough_multitask.pt'} or {assets_path / 'cough_model.pt'}. "
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
                if 'model_state_dict' in checkpoint:
                    # State dict format
                    if self.model_class is None:
                        raise ValueError("model_class required when loading state dict. Provide the model architecture class.")
                    self.model = self.model_class()
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model = self.model_class() if self.model_class else None
                    if self.model:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        raise ValueError("model_class required when loading state dict")
                else:
                    # Assume it's a state dict directly
                    if self.model_class is None:
                        # Try to use default architecture
                        logger.info("No model_class provided, using default SimpleCoughModel architecture")
                        self.model_class = SimpleCoughModel
                    self.model = self.model_class(num_outputs=self.num_outputs)
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
            
            logger.info(f"Loaded cough model from {self.model_path}")
            logger.info(f"Model device: {self.device}, outputs: {self.num_outputs}")
        except Exception as e:
            logger.error(f"Error loading cough VAD model: {e}", exc_info=True)
            self.model = None
    
    def predict(self, features: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Predict cough and attribute probabilities.
        
        Args:
            features: Feature array (1s log-Mel spectrogram)
        
        Returns:
            Tuple of (p_cough, p_attr_wet, p_attr_stridor, p_attr_choking, p_attr_congestion, p_attr_wheezing_selfreport)
            All values in [0, 1]
        """
        if self.model is None:
            # Mock prediction for development
            logger.warning("Using mock cough model prediction (model not loaded)")
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
            
            # Convert to numpy and extract probabilities
            if isinstance(output, torch.Tensor):
                output_array = output.cpu().numpy()
            else:
                output_array = np.array(output)
            
            # Flatten if needed
            if len(output_array.shape) > 1:
                output_array = output_array.flatten()
            
            # Ensure we have 6 outputs
            if len(output_array) < self.num_outputs:
                logger.warning(f"Model output has {len(output_array)} values, expected {self.num_outputs}")
                output_array = np.pad(output_array, (0, self.num_outputs - len(output_array)), mode='constant')
            elif len(output_array) > self.num_outputs:
                output_array = output_array[:self.num_outputs]
            
            # Apply sigmoid if needed (assuming raw logits)
            # Uncomment if your model outputs logits instead of probabilities
            # output_array = 1 / (1 + np.exp(-output_array))
            
            # Extract probabilities and ensure valid range
            p_cough = float(np.clip(output_array[0], 0.0, 1.0))
            p_attr_wet = float(np.clip(output_array[1], 0.0, 1.0))
            p_attr_stridor = float(np.clip(output_array[2], 0.0, 1.0))
            p_attr_choking = float(np.clip(output_array[3], 0.0, 1.0))
            p_attr_congestion = float(np.clip(output_array[4], 0.0, 1.0))
            p_attr_wheezing_selfreport = float(np.clip(output_array[5], 0.0, 1.0))
            
            logger.debug(f"Cough model prediction: cough={p_cough:.3f}, wet={p_attr_wet:.3f}, stridor={p_attr_stridor:.3f}")
            return (p_cough, p_attr_wet, p_attr_stridor, p_attr_choking, p_attr_congestion, p_attr_wheezing_selfreport)
            
        except Exception as e:
            logger.error(f"Error in cough model prediction: {e}", exc_info=True)
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
    
    def _mock_predict(self, features: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Mock prediction for development/testing."""
        # Simple heuristic: if audio has high energy in mid frequencies, might be cough
        if len(features.shape) >= 2:
            mid_freq_energy = np.mean(features[:, features.shape[1]//4:3*features.shape[1]//4])
            p_cough = min(0.8, max(0.2, mid_freq_energy))
        else:
            p_cough = 0.5
        
        # Mock attribute probabilities (lower than cough probability)
        p_attr_wet = p_cough * 0.6
        p_attr_stridor = p_cough * 0.3
        p_attr_choking = p_cough * 0.2
        p_attr_congestion = p_cough * 0.5
        p_attr_wheezing_selfreport = p_cough * 0.4
        
        return (p_cough, p_attr_wet, p_attr_stridor, p_attr_choking, p_attr_congestion, p_attr_wheezing_selfreport)
