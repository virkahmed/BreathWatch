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
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger(__name__)

ATTRIBUTE_KEYS = ["wet", "wheezing", "stridor", "choking", "congestion"]


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
    
    def __init__(self, model_path: Optional[str] = None, model_class: Optional[callable] = None):
        """
        Initialize cough model.
        
        Args:
            model_path: Path to PyTorch model file (.pt). If None, looks in assets folder.
            model_class: Optional PyTorch model class to instantiate. Required if loading state dict.
        """
        self.model = None
        self.model_path = model_path
        self.model_class = model_class
        self.input_shape = (1, 256, 256)  # Matches CoughMultitaskCNN training specs
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
                        raise ValueError("model_class required when loading state dict. Provide a callable that returns the model.")
                    self.model = self.model_class()  # Call the factory function
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    if self.model_class is None:
                        raise ValueError("model_class required when loading state dict")
                    self.model = self.model_class()  # Call the factory function
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume it's a state dict directly
                    if self.model_class is None:
                        # Try to use default architecture
                        logger.info("No model_class provided, using default SimpleCoughModel architecture")
                        self.model = SimpleCoughModel(num_outputs=self.num_outputs)
                    else:
                        self.model = self.model_class()  # Call the factory function
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
    
    def predict(self, features: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Predict cough probability and attribute probability vector.
        
        Args:
            features: Feature array (1s log-Mel spectrogram)
        
        Returns:
            Tuple of (p_cough, {"wet": ..., "wheezing": ..., ...})
        """
        if self.model is None:
            logger.warning("Using mock cough model prediction (model not loaded)")
            return self._mock_predict(features)
        
        try:
            input_tensor = self._prepare_input(features)
            with torch.no_grad():
                output = self.model(input_tensor)
            
            if isinstance(output, (list, tuple)):
                logits_np = self._to_numpy(output[0])
                p_cough = self._extract_cough_probability(logits_np)
                
                if len(output) > 1:
                    attr_values = self._extract_attribute_probs(output[1])
                else:
                    attr_values = self._infer_attributes_from_logits(logits_np)
            else:
                flat_output = self._to_numpy(output)
                p_cough, attr_values = self._decode_flat_output(flat_output)
            
            p_cough_clamped, attr_dict = self._format_prediction(p_cough, attr_values)
            logger.debug(f"Cough model prediction: p_cough={p_cough_clamped:.3f}, attrs={attr_dict}")
            return p_cough_clamped, attr_dict
        
        except Exception as e:
            logger.error(f"Error in cough model prediction: {e}", exc_info=True)
            return self._mock_predict(features)
    
    def _to_numpy(self, value) -> np.ndarray:
        """Convert torch tensors or sequences to numpy arrays."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.array(value)
    
    def _extract_cough_probability(self, logits_np: np.ndarray) -> float:
        """Extract cough probability from logits or probabilities."""
        if logits_np is None:
            return 0.0
        flat = logits_np.reshape(-1)
        if flat.size >= 2:
            logits_pair = flat[:2]
            exp_logits = np.exp(logits_pair - np.max(logits_pair))
            probs = exp_logits / np.sum(exp_logits)
            return float(np.clip(probs[1], 0.0, 1.0))
        elif flat.size == 1:
            value = flat[0]
            if value < 0.0 or value > 1.0:
                value = 1 / (1 + np.exp(-np.clip(value, -500, 500)))
            return float(np.clip(value, 0.0, 1.0))
        return 0.0
    
    def _extract_attribute_probs(self, attr_output) -> np.ndarray:
        """Normalize attribute outputs to probabilities in [0,1]."""
        attr_np = self._to_numpy(attr_output)
        if attr_np.ndim > 1:
            attr_vec = attr_np.reshape(attr_np.shape[0], -1)[0]
        else:
            attr_vec = attr_np.flatten()
        
        if attr_vec.size < len(ATTRIBUTE_KEYS):
            attr_vec = np.pad(attr_vec, (0, len(ATTRIBUTE_KEYS) - attr_vec.size), mode="constant")
        else:
            attr_vec = attr_vec[:len(ATTRIBUTE_KEYS)]
        
        if np.any(attr_vec < 0.0) or np.any(attr_vec > 1.0):
            attr_vec = 1 / (1 + np.exp(-np.clip(attr_vec, -500, 500)))
        else:
            attr_vec = np.clip(attr_vec, 0.0, 1.0)
        return attr_vec
    
    def _infer_attributes_from_logits(self, logits_np: np.ndarray) -> np.ndarray:
        """Infer attribute probabilities from logits tensor when no explicit attr head is present."""
        if logits_np is None:
            return np.zeros(len(ATTRIBUTE_KEYS))
        
        if logits_np.ndim >= 2 and logits_np.shape[1] >= len(ATTRIBUTE_KEYS) + 2:
            attr_logits = logits_np[0, 2:2 + len(ATTRIBUTE_KEYS)]
        else:
            flat = logits_np.reshape(-1)
            if flat.size >= len(ATTRIBUTE_KEYS) + 2:
                attr_logits = flat[2:2 + len(ATTRIBUTE_KEYS)]
            elif flat.size >= len(ATTRIBUTE_KEYS) + 1:
                attr_logits = flat[1:1 + len(ATTRIBUTE_KEYS)]
            else:
                return np.zeros(len(ATTRIBUTE_KEYS))
        return 1 / (1 + np.exp(-np.clip(attr_logits, -500, 500)))
    
    def _decode_flat_output(self, flat_output: np.ndarray) -> Tuple[float, np.ndarray]:
        """Handle single-tensor outputs that include cough + attributes."""
        if flat_output is None:
            return 0.0, np.zeros(len(ATTRIBUTE_KEYS))
        
        flat = flat_output.flatten()
        if flat.size >= len(ATTRIBUTE_KEYS) + 2:
            p_cough = self._extract_cough_probability(flat[:2])
            attr_values = self._extract_attribute_probs(flat[2:2 + len(ATTRIBUTE_KEYS)])
            return p_cough, attr_values
        
        if flat.size == 0:
            return 0.0, np.zeros(len(ATTRIBUTE_KEYS))
        
        p_cough_val = flat[0]
        attr_slice = flat[1:] if flat.size > 1 else np.zeros(0)
        attr_values = self._extract_attribute_probs(attr_slice)
        if p_cough_val < 0.0 or p_cough_val > 1.0:
            p_cough_val = 1 / (1 + np.exp(-np.clip(p_cough_val, -500, 500)))
        return float(np.clip(p_cough_val, 0.0, 1.0)), attr_values
    
    def _format_prediction(self, p_cough: float, attr_values: Optional[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        """Clamp outputs and convert to attribute dictionary."""
        attr_list = list(attr_values.flatten()) if attr_values is not None else []
        attr_dict: Dict[str, float] = {}
        for idx, key in enumerate(ATTRIBUTE_KEYS):
            value = attr_list[idx] if idx < len(attr_list) else 0.0
            attr_dict[key] = float(np.clip(value, 0.0, 1.0))
        return float(np.clip(p_cough, 0.0, 1.0)), attr_dict
    
    def _prepare_input(self, features: np.ndarray) -> torch.Tensor:
        """
        Prepare features for model input.
        
        Expects features in shape [1, 256, 256] from prepare_mobile_features.
        Converts to [batch_size, 1, 256, 256] for model.
        """
        # Features should already be [1, 256, 256] from prepare_mobile_features
        # But handle different input shapes gracefully
        
        # If 2D, add channel dimension
        if len(features.shape) == 2:
            features = np.expand_dims(features, axis=0)
        
        # If shape is [1, H, W] but not [1, 256, 256], resize
        if len(features.shape) == 3 and features.shape[1:] != self.input_shape[1:]:
            from scipy.ndimage import zoom
            current_shape = features.shape[1:]
            zoom_factors = (
                self.input_shape[1] / current_shape[0],  # Height
                self.input_shape[2] / current_shape[1]   # Width
            )
            features = zoom(features, (1,) + zoom_factors, order=1)
        
        # Add batch dimension if missing (should be [batch, 1, 256, 256])
        if len(features.shape) == 3:
            features = np.expand_dims(features, axis=0)
        
        # Ensure shape is [batch, 1, 256, 256]
        if features.shape[1:] != self.input_shape:
            logger.warning(f"Feature shape {features.shape} doesn't match expected {self.input_shape}, attempting resize")
            from scipy.ndimage import zoom
            if len(features.shape) == 4:
                # [batch, channels, H, W]
                zoom_factors = (
                    1,  # batch
                    1,  # channels
                    self.input_shape[1] / features.shape[2],  # height
                    self.input_shape[2] / features.shape[3]   # width
                )
                features = zoom(features, zoom_factors, order=1)
        
        # Convert to torch tensor
        input_tensor = torch.from_numpy(features.astype(np.float32))
        input_tensor = input_tensor.to(self.device)
        
        return input_tensor
    
    def _mock_predict(self, features: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Mock prediction for development/testing."""
        # Simple heuristic: if audio has high energy in mid frequencies, might be cough
        if len(features.shape) >= 2:
            mid_freq_energy = np.mean(features[:, features.shape[1]//4:3*features.shape[1]//4])
            p_cough = min(0.8, max(0.2, mid_freq_energy))
        else:
            p_cough = 0.5
        
        attr_dict = {
            "wet": float(np.clip(p_cough * 0.6, 0.0, 1.0)),
            "wheezing": float(np.clip(p_cough * 0.4, 0.0, 1.0)),
            "stridor": float(np.clip(p_cough * 0.3, 0.0, 1.0)),
            "choking": float(np.clip(p_cough * 0.2, 0.0, 1.0)),
            "congestion": float(np.clip(p_cough * 0.5, 0.0, 1.0)),
        }
        
        return float(np.clip(p_cough, 0.0, 1.0)), attr_dict
