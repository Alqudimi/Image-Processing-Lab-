import cv2
import numpy as np
from typing import Union, List, Dict, Any, Optional, Callable
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilterType(Enum):
    GAUSSIAN_BLUR = "gaussian_blur"
    MEDIAN_BLUR = "median_blur"
    BILATERAL_FILTER = "bilateral_filter"
    SOBEL = "sobel"
    CANNY = "canny"
    LAPLACIAN = "laplacian"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    GAMMA_CORRECTION = "gamma_correction"
    THRESHOLD = "threshold"
    ADAPTIVE_THRESHOLD = "adaptive_threshold"
    MORPHOLOGICAL = "morphological"
    CUSTOM = "custom"

class ImageProcessingException(Exception):
    pass

class InvalidImageException(ImageProcessingException):
    pass

class InvalidParameterException(ImageProcessingException):
    pass

class FilterConfigurationException(ImageProcessingException):
    pass

class AdvancedImageProcessor:
    def __init__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise InvalidImageException("Input must be a numpy array")
        if image.size == 0:
            raise InvalidImageException("Image cannot be empty")
        
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.history = []
        self.filter_configs = {}
        
    def validate_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise InvalidImageException("Image must be a numpy array")
        if image.size == 0:
            raise InvalidImageException("Image cannot be empty")
        if len(image.shape) not in [2, 3]:
            raise InvalidImageException("Image must be 2D or 3D array")

    def validate_kernel_size(self, ksize: int) -> None:
        if not isinstance(ksize, int) or ksize <= 0 or ksize % 2 == 0:
            raise InvalidParameterException("Kernel size must be positive odd integer")

    def validate_parameters(self, params: Dict[str, Any], required_params: List[str]) -> None:
        for param in required_params:
            if param not in params:
                raise InvalidParameterException(f"Missing required parameter: {param}")

    def apply_filter(self, filter_type: Union[FilterType, str], **kwargs) -> np.ndarray:
        try:
            if isinstance(filter_type, str):
                filter_type = FilterType(filter_type.lower())
            
            filter_methods = {
                FilterType.GAUSSIAN_BLUR: self._apply_gaussian_blur,
                FilterType.MEDIAN_BLUR: self._apply_median_blur,
                FilterType.BILATERAL_FILTER: self._apply_bilateral_filter,
                FilterType.SOBEL: self._apply_sobel,
                FilterType.CANNY: self._apply_canny,
                FilterType.LAPLACIAN: self._apply_laplacian,
                FilterType.HISTOGRAM_EQUALIZATION: self._apply_histogram_equalization,
                FilterType.GAMMA_CORRECTION: self._apply_gamma_correction,
                FilterType.THRESHOLD: self._apply_threshold,
                FilterType.ADAPTIVE_THRESHOLD: self._apply_adaptive_threshold,
                FilterType.MORPHOLOGICAL: self._apply_morphological,
                FilterType.CUSTOM: self._apply_custom_filter
            }
            
            if filter_type not in filter_methods:
                raise FilterConfigurationException(f"Unsupported filter type: {filter_type}")
            
            result = filter_methods[filter_type](**kwargs)
            self.history.append((filter_type, kwargs))
            self.current_image = result
            return result
            
        except Exception as e:
            logger.error(f"Error applying filter {filter_type}: {str(e)}")
            raise ImageProcessingException(f"Failed to apply filter: {str(e)}")

    def _apply_gaussian_blur(self, ksize: int = 5, sigma: float = 0) -> np.ndarray:
        self.validate_kernel_size(ksize)
        if sigma < 0:
            raise InvalidParameterException("Sigma must be non-negative")
        return cv2.GaussianBlur(self.current_image, (ksize, ksize), sigma)

    def _apply_median_blur(self, ksize: int = 5) -> np.ndarray:
        self.validate_kernel_size(ksize)
        return cv2.medianBlur(self.current_image, ksize)

    def _apply_bilateral_filter(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        if d <= 0 or sigma_color <= 0 or sigma_space <= 0:
            raise InvalidParameterException("Parameters must be positive")
        return cv2.bilateralFilter(self.current_image, d, sigma_color, sigma_space)

    def _apply_sobel(self, dx: int = 1, dy: int = 0, ksize: int = 3, scale: float = 1.0, delta: float = 0.0) -> np.ndarray:
        self.validate_kernel_size(ksize)
        if dx < 0 or dy < 0:
            raise InvalidParameterException("dx and dy must be non-negative")
        
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize, scale=scale, delta=delta)
        return np.uint8(np.absolute(sobel))

    def _apply_canny(self, threshold1: float = 100, threshold2: float = 200, aperture_size: int = 3) -> np.ndarray:
        if threshold1 <= 0 or threshold2 <= 0:
            raise InvalidParameterException("Threshold must be positive")
        if aperture_size not in [3, 5, 7]:
            raise InvalidParameterException("Aperture size must be 3, 5, or 7")
        
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        return cv2.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)

    def _apply_laplacian(self, ksize: int = 3, scale: float = 1.0, delta: float = 0.0) -> np.ndarray:
        self.validate_kernel_size(ksize)
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale, delta=delta)
        return np.uint8(np.absolute(laplacian))

    def _apply_histogram_equalization(self) -> np.ndarray:
        if len(self.current_image.shape) == 3:
            hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            return cv2.equalizeHist(self.current_image)

    def _apply_gamma_correction(self, gamma: float = 1.0) -> np.ndarray:
        if gamma <= 0:
            raise InvalidParameterException("Gamma must be positive")
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.current_image, table)

    def _apply_threshold(self, thresh: float = 127, maxval: float = 255, type: int = cv2.THRESH_BINARY) -> np.ndarray:
        if thresh < 0 or maxval < 0:
            raise InvalidParameterException("Threshold value must be non-negative")
        
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        _, thresholded = cv2.threshold(gray, thresh, maxval, type)
        return thresholded

    def _apply_adaptive_threshold(self, max_value: float = 255, adaptive_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 threshold_type: int = cv2.THRESH_BINARY, block_size: int = 11, C: float = 2) -> np.ndarray:
        if block_size % 2 == 0:
            raise InvalidParameterException("Block size must be odd")
        if max_value <= 0:
            raise InvalidParameterException("Max value must be positive")
        
        if len(self.current_image.shape) == 3:
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.current_image
            
        return cv2.adaptiveThreshold(gray, max_value, adaptive_method, threshold_type, block_size, C)

    def _apply_morphological(self, operation: int, kernel_size: int = 3, 
                           iterations: int = 1, kernel_shape: int = cv2.MORPH_RECT) -> np.ndarray:
        self.validate_kernel_size(kernel_size)
        if iterations <= 0:
            raise InvalidParameterException("Iterations must be positive")
        
        kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        return cv2.morphologyEx(self.current_image, operation, kernel, iterations=iterations)

    def _apply_custom_filter(self, kernel: np.ndarray, delta: float = 0.0) -> np.ndarray:
        if not isinstance(kernel, np.ndarray):
            raise InvalidParameterException("Kernel must be a numpy array")
        if kernel.size == 0:
            raise InvalidParameterException("Kernel cannot be empty")
        
        return cv2.filter2D(self.current_image, -1, kernel, delta=delta)

    def apply_filter_chain(self, filter_chain: List[Dict[str, Any]]) -> np.ndarray:
        try:
            for filter_config in filter_chain:
                filter_type = filter_config.get("filter_type")
                params = filter_config.get("parameters", {})
                self.apply_filter(filter_type, **params)
            return self.current_image
        except Exception as e:
            logger.error(f"Error in filter chain: {str(e)}")
            raise ImageProcessingException(f"Filter chain failed: {str(e)}")

    def reset_to_original(self) -> None:
        self.current_image = self.original_image.copy()
        self.history.clear()

    def undo_last_filter(self) -> Optional[np.ndarray]:
        if not self.history:
            return None
        
        self.history.pop()
        self.reset_to_original()
        
        for filter_type, params in self.history:
            self.apply_filter(filter_type, **params)
        
        return self.current_image

    def get_current_image(self) -> np.ndarray:
        return self.current_image.copy()

    def get_original_image(self) -> np.ndarray:
        return self.original_image.copy()

    def get_history(self) -> List[tuple]:
        return self.history.copy()

    def save_current_image(self, filepath: str) -> bool:
        try:
            success = cv2.imwrite(filepath, self.current_image)
            if not success:
                raise ImageProcessingException(f"Failed to save image to {filepath}")
            return success
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            raise ImageProcessingException(f"Failed to save image: {str(e)}")

    def set_custom_filter_config(self, filter_name: str, config: Dict[str, Any]) -> None:
        self.filter_configs[filter_name] = config

    def get_custom_filter_config(self, filter_name: str) -> Optional[Dict[str, Any]]:
        return self.filter_configs.get(filter_name)

    def batch_process(self, images: List[np.ndarray], filter_chain: List[Dict[str, Any]]) -> List[np.ndarray]:
        results = []
        for img in images:
            try:
                processor = AdvancedImageProcessor(img)
                result = processor.apply_filter_chain(filter_chain)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing image in batch: {str(e)}")
                raise ImageProcessingException(f"Batch processing failed: {str(e)}")
        return results


