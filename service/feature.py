import cv2
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureType(Enum):
    FAST_CORNERS = "fast_corners"
    HOG = "hog"
    LOG_DOG_BLOB = "log_dog_blob"
    ORB = "orb"
    SIFT = "sift"

class FeatureExtractionException(Exception):
    pass

class InvalidImageException(FeatureExtractionException):
    pass

class InvalidParameterException(FeatureExtractionException):
    pass

class FeatureConfigurationException(FeatureExtractionException):
    pass

@dataclass
class FeatureResult:
    keypoints: Optional[List[cv2.KeyPoint]]
    descriptors: Optional[np.ndarray]
    features: Optional[Dict[str, Any]]
    image: Optional[np.ndarray]
    metadata: Dict[str, Any]

class AdvancedFeatureExtractor:
    def __init__(self, image: np.ndarray):
        self._validate_image(image)
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.feature_history = []
        self.extractor_configs = {}
        
    def _validate_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise InvalidImageException("الصورة يجب أن تكون مصفوفة numpy")
        if image.size == 0:
            raise InvalidImageException("الصورة لا يمكن أن تكون فارغة")
        if len(image.shape) not in [2, 3]:
            raise InvalidImageException("الصورة يجب أن تكون ثنائية أو ثلاثية الأبعاد")

    def _validate_positive_int(self, value: int, param_name: str) -> None:
        if not isinstance(value, int) or value <= 0:
            raise InvalidParameterException(f"{param_name} يجب أن يكون عدد صحيح موجب")

    def _validate_positive_float(self, value: float, param_name: str) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise InvalidParameterException(f"{param_name} يجب أن يكون عدد حقيقي موجب")

    def _validate_parameters(self, params: Dict[str, Any], required_params: List[str]) -> None:
        for param in required_params:
            if param not in params:
                raise InvalidParameterException(f"معلمة مطلوبة مفقودة: {param}")

    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _draw_keypoints(self, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        return cv2.drawKeypoints(self.current_image, keypoints, None, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    def extract_features(self, feature_type: Union[FeatureType, str], **kwargs) -> FeatureResult:
        try:
            if isinstance(feature_type, str):
                feature_type = FeatureType(feature_type.lower())
            
            extraction_methods = {
                FeatureType.FAST_CORNERS: self._extract_fast_corners,
                FeatureType.HOG: self._extract_hog,
                FeatureType.LOG_DOG_BLOB: self._extract_log_dog_blob,
                FeatureType.ORB: self._extract_orb,
                FeatureType.SIFT: self._extract_sift
            }
            
            if feature_type not in extraction_methods:
                raise FeatureConfigurationException(f"نوع الميزة غير مدعوم: {feature_type}")
            
            result = extraction_methods[feature_type](**kwargs)
            self.feature_history.append((feature_type, kwargs, result.metadata))
            return result
            
        except Exception as e:
            logger.error(f"خطأ في استخراج الميزات {feature_type}: {str(e)}")
            raise FeatureExtractionException(f"فشل في استخراج الميزات: {str(e)}")

    def _extract_fast_corners(self, threshold: int = 10, nonmax_suppression: bool = True, 
                            type: int = cv2.FAST_FEATURE_DETECTOR_TYPE_9_16) -> FeatureResult:
        try:
            self._validate_positive_int(threshold, "threshold")
            
            fast = cv2.FastFeatureDetector_create(threshold=threshold,
                                                nonmaxSuppression=nonmax_suppression,
                                                type=type)
            
            gray = self._convert_to_grayscale(self.current_image)
            keypoints = fast.detect(gray, None)
            
            return FeatureResult(
                keypoints=keypoints,
                descriptors=None,
                features={'keypoints_count': len(keypoints), 'corners_detected': len(keypoints)},
                image=self._draw_keypoints(keypoints),
                metadata={'method': 'FAST_CORNERS', 'parameters': locals()}
            )
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في FAST Corners: {str(e)}")

    def _extract_hog(self, win_size: tuple = (64, 128), block_size: tuple = (16, 16), 
                    block_stride: tuple = (8, 8), cell_size: tuple = (8, 8), 
                    nbins: int = 9, deriv_aperture: int = 1, win_sigma: float = 4.0, 
                    histogram_norm_type: int = 0, l2_hys_threshold: float = 2.0, 
                    gamma_correction: bool = True, nlevels: int = 64) -> FeatureResult:
        try:
            self._validate_positive_int(nbins, "nbins")
            self._validate_positive_float(win_sigma, "win_sigma")
            
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, 
                                  nbins, deriv_aperture, win_sigma, histogram_norm_type,
                                  l2_hys_threshold, gamma_correction, nlevels)
            
            gray = self._convert_to_grayscale(self.current_image)
            gray_resized = cv2.resize(gray, win_size)
            
            hog_features = hog.compute(gray_resized)
            hog_image = self._visualize_hog(gray_resized, hog, cell_size)
            
            return FeatureResult(
                keypoints=None,
                descriptors=hog_features,
                features={'feature_vector_length': len(hog_features), 'descriptor_shape': hog_features.shape},
                image=hog_image,
                metadata={'method': 'HOG', 'parameters': locals()}
            )
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في HOG: {str(e)}")

    def _visualize_hog(self, image: np.ndarray, hog: cv2.HOGDescriptor, cell_size: tuple) -> np.ndarray:
        try:
            hog_image = np.zeros_like(image)
            features = hog.compute(image)
            
            if len(features) > 0:
                height, width = image.shape
                n_cells_x = width // cell_size[0]
                n_cells_y = height // cell_size[1]
                
                max_len = cell_size[0] // 2
                hog_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                for i in range(n_cells_y):
                    for j in range(n_cells_x):
                        cell_features = features[i * n_cells_x + j * 9:(i * n_cells_x + j + 1) * 9]
                        if len(cell_features) == 9:
                            center = (j * cell_size[0] + cell_size[0] // 2, 
                                    i * cell_size[1] + cell_size[1] // 2)
                            
                            for k in range(9):
                                angle = k * 20
                                rad = np.deg2rad(angle)
                                magnitude = cell_features[k] * max_len
                                
                                x2 = int(center[0] + magnitude * np.cos(rad))
                                y2 = int(center[1] + magnitude * np.sin(rad))
                                
                                cv2.line(hog_image, center, (x2, y2), (0, 255, 0), 1)
            
            return hog_image
        except:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def _extract_log_dog_blob(self, min_threshold: float = 10.0, max_threshold: float = 200.0, 
                             threshold_step: float = 10.0, min_area: float = 100.0, 
                             min_circularity: float = 0.8, min_convexity: float = 0.9, 
                             min_inertia_ratio: float = 0.5) -> FeatureResult:
        try:
            self._validate_positive_float(min_threshold, "min_threshold")
            self._validate_positive_float(max_threshold, "max_threshold")
            
            params = cv2.SimpleBlobDetector_Params()
            params.minThreshold = min_threshold
            params.maxThreshold = max_threshold
            params.thresholdStep = threshold_step
            params.filterByArea = True
            params.minArea = min_area
            params.filterByCircularity = True
            params.minCircularity = min_circularity
            params.filterByConvexity = True
            params.minConvexity = min_convexity
            params.filterByInertia = True
            params.minInertiaRatio = min_inertia_ratio
            
            detector = cv2.SimpleBlobDetector_create(params)
            gray = self._convert_to_grayscale(self.current_image)
            keypoints = detector.detect(gray)
            
            return FeatureResult(
                keypoints=keypoints,
                descriptors=None,
                features={'blobs_count': len(keypoints), 'keypoints_count': len(keypoints)},
                image=self._draw_keypoints(keypoints),
                metadata={'method': 'LOG_DOG_BLOB', 'parameters': locals()}
            )
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في LoG/DoG Blob Detection: {str(e)}")

    def _extract_orb(self, n_features: int = 500, scale_factor: float = 1.2, 
                    n_levels: int = 8, edge_threshold: int = 31, 
                    first_level: int = 0, wta_k: int = 2, 
                    score_type: int = cv2.ORB_HARRIS_SCORE, patch_size: int = 31) -> FeatureResult:
        try:
            self._validate_positive_int(n_features, "n_features")
            self._validate_positive_float(scale_factor, "scale_factor")
            
            orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor,
                                nlevels=n_levels, edgeThreshold=edge_threshold,
                                firstLevel=first_level, WTA_K=wta_k,
                                scoreType=score_type, patchSize=patch_size)
            
            gray = self._convert_to_grayscale(self.current_image)
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            return FeatureResult(
                keypoints=keypoints,
                descriptors=descriptors,
                features={'keypoints_count': len(keypoints), 'descriptors_shape': descriptors.shape if descriptors is not None else None},
                image=self._draw_keypoints(keypoints),
                metadata={'method': 'ORB', 'parameters': locals()}
            )
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في ORB: {str(e)}")

    def _extract_sift(self, n_features: int = 0, n_octave_layers: int = 3, 
                     contrast_threshold: float = 0.04, edge_threshold: float = 10, 
                     sigma: float = 1.6) -> FeatureResult:
        try:
            self._validate_positive_int(n_octave_layers, "n_octave_layers")
            self._validate_positive_float(contrast_threshold, "contrast_threshold")
            
            if n_features == 0:
                sift = cv2.SIFT_create(nOctaveLayers=n_octave_layers, 
                                      contrastThreshold=contrast_threshold,
                                      edgeThreshold=edge_threshold, sigma=sigma)
            else:
                sift = cv2.SIFT_create(nfeatures=n_features, nOctaveLayers=n_octave_layers,
                                      contrastThreshold=contrast_threshold,
                                      edgeThreshold=edge_threshold, sigma=sigma)
            
            gray = self._convert_to_grayscale(self.current_image)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            
            return FeatureResult(
                keypoints=keypoints,
                descriptors=descriptors,
                features={'keypoints_count': len(keypoints), 'descriptors_shape': descriptors.shape if descriptors is not None else None},
                image=self._draw_keypoints(keypoints),
                metadata={'method': 'SIFT', 'parameters': locals()}
            )
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في SIFT: {str(e)}")

    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray, 
                      method: str = "BF", distance_type: str = "L2", 
                      ratio_test: float = 0.75, k: int = 2) -> List[cv2.DMatch]:
        try:
            if descriptors1 is None or descriptors2 is None:
                raise InvalidParameterException("المقومات غير متوفرة للمطابقة")
            
            if method.upper() == "BF":
                if distance_type.upper() == "L2":
                    matcher = cv2.BFMatcher(cv2.NORM_L2)
                elif distance_type.upper() == "HAMMING":
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
                else:
                    raise InvalidParameterException("نوع المسافة غير مدعوم")
            else:
                raise InvalidParameterException("طريقة المطابقة غير مدعومة")
            
            matches = matcher.knnMatch(descriptors1, descriptors2, k=k)
            
            if ratio_test > 0:
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_test * n.distance:
                            good_matches.append(m)
                return good_matches
            return [m[0] for m in matches if len(m) > 0]
            
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في مطابقة الميزات: {str(e)}")

    def draw_matches(self, image1: np.ndarray, keypoints1: List[cv2.KeyPoint],
                    image2: np.ndarray, keypoints2: List[cv2.KeyPoint],
                    matches: List[cv2.DMatch], matches_thickness: int = 2) -> np.ndarray:
        try:
            return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                                 matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                 matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        except Exception as e:
            raise FeatureExtractionException(f"خطأ في رسم المطابقات: {str(e)}")

    def get_feature_history(self) -> List[Tuple]:
        return self.feature_history.copy()

    def reset_to_original(self) -> None:
        self.current_image = self.original_image.copy()
        self.feature_history.clear()

    def set_current_image(self, image: np.ndarray) -> None:
        self._validate_image(image)
        self.current_image = image.copy()

    def get_current_image(self) -> np.ndarray:
        return self.current_image.copy()

    def get_original_image(self) -> np.ndarray:
        return self.original_image.copy()

    def set_extractor_config(self, config_name: str, config: Dict[str, Any]) -> None:
        self.extractor_configs[config_name] = config

    def get_extractor_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        return self.extractor_configs.get(config_name)

    def batch_extract_features(self, images: List[np.ndarray], feature_type: FeatureType, **kwargs) -> List[FeatureResult]:
        results = []
        for img in images:
            try:
                extractor = AdvancedFeatureExtractor(img)
                result = extractor.extract_features(feature_type, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"خطأ في المعالجة الدفعية: {str(e)}")
                raise FeatureExtractionException(f"فشل في المعالجة الدفعية: {str(e)}")
        return results


