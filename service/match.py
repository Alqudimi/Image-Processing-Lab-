import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional, Union

class MatchingMethod(Enum):
    FLANN = "FLANN"
    BF_SIFT_RATIO = "BF_SIFT_RATIO"
    BF_ORB_RATIO = "BF_ORB_RATIO"

class FeatureMatching:
    def __init__(self, image1: np.ndarray, image2: np.ndarray):
        if image1 is None or image2 is None:
            raise ValueError("الصور المدخلة لا يمكن أن تكون None")
        
        if not isinstance(image1, np.ndarray) or not isinstance(image2, np.ndarray):
            raise TypeError("نوع الصور يجب أن يكون numpy.ndarray")
        
        if image1.size == 0 or image2.size == 0:
            raise ValueError("الصور المدخلة فارغة")
        
        if len(image1.shape) == 3:
            self.image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            self.image1 = image1.copy()
            
        if len(image2.shape) == 3:
            self.image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            self.image2 = image2.copy()
            
        self.keypoints1 = None
        self.keypoints2 = None
        self.descriptors1 = None
        self.descriptors2 = None
        self.matches = None
        self.good_matches = None
        self.homography = None
        self.detector = None
        self.matcher = None
        self.matching_method = None
        
    def _validate_image(self) -> None:
        if self.image1 is None or self.image2 is None:
            raise ValueError("الصور غير موجودة")
    
    def _validate_features_detected(self) -> None:
        if self.keypoints1 is None or self.descriptors1 is None:
            raise RuntimeError("يجب اكتشاف الميزات أولاً باستخدام detect_features()")
    
    def _validate_matches_calculated(self) -> None:
        if self.matches is None:
            raise RuntimeError("يجب حساب المطابقات أولاً باستخدام match_features()")
    
    def detect_features(self, method: MatchingMethod) -> 'FeatureMatching':
        try:
            self._validate_image()
            self.matching_method = method
            
            if method == MatchingMethod.FLANN or method == MatchingMethod.BF_SIFT_RATIO:
                self.detector = cv2.SIFT_create()
            elif method == MatchingMethod.BF_ORB_RATIO:
                self.detector = cv2.ORB_create(nfeatures=1000)
            else:
                raise ValueError(f"طريقة المطابقة {method} غير مدعومة")
            
            self.keypoints1, self.descriptors1 = self.detector.detectAndCompute(self.image1, None)
            self.keypoints2, self.descriptors2 = self.detector.detectAndCompute(self.image2, None)
            
            if len(self.keypoints1) == 0 or len(self.keypoints2) == 0:
                raise RuntimeError("لم يتم اكتشاف أي ميزات في إحدى الصور أو كلتيهما")
            
            if self.descriptors1 is None or self.descriptors2 is None:
                raise RuntimeError("فشل في حساب الواصفات للميزات")
            
            if method == MatchingMethod.FLANN:
                self.descriptors1 = self.descriptors1.astype(np.float32)
                self.descriptors2 = self.descriptors2.astype(np.float32)
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في اكتشاف الميزات: {str(e)}")
    
    def match_features(self, ratio_threshold: float = 0.75, max_distance: float = 100.0) -> 'FeatureMatching':
        try:
            self._validate_features_detected()
            
            if self.matching_method == MatchingMethod.FLANN:
                index_params = dict(algorithm=1, trees=5)
                search_params = dict(checks=50)
                self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
                self.matches = self.matcher.knnMatch(self.descriptors1, self.descriptors2, k=2)
                
            elif self.matching_method == MatchingMethod.BF_SIFT_RATIO:
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                self.matches = self.matcher.knnMatch(self.descriptors1, self.descriptors2, k=2)
                
            elif self.matching_method == MatchingMethod.BF_ORB_RATIO:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                self.matches = self.matcher.knnMatch(self.descriptors1, self.descriptors2, k=2)
            
            else:
                raise ValueError(f"طريقة المطابقة {self.matching_method} غير مدعومة")
            
            self.good_matches = []
            
            if self.matching_method in [MatchingMethod.FLANN, MatchingMethod.BF_SIFT_RATIO, MatchingMethod.BF_ORB_RATIO]:
                for match_pair in self.matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_threshold * n.distance:
                            self.good_matches.append(m)
            
            if len(self.good_matches) == 0:
                raise RuntimeError("لم يتم العثور على أي مطابقات جيدة بعد التصفية")
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في مطابقة الميزات: {str(e)}")
    
    def calculate_homography(self, ransac_thresh: float = 5.0) -> 'FeatureMatching':
        try:
            self._validate_matches_calculated()
            
            if len(self.good_matches) < 4:
                raise RuntimeError("يحتاج حساب homography إلى 4 مطابقات على الأقل")
            
            src_pts = np.float32([self.keypoints1[m.queryIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.keypoints2[m.trainIdx].pt for m in self.good_matches]).reshape(-1, 1, 2)
            
            self.homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
            
            if self.homography is None:
                raise RuntimeError("فشل في حساب homography")
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في حساب homography: {str(e)}")
    
    def draw_matches(self, output_image: Optional[np.ndarray] = None, 
                    flags: int = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) -> np.ndarray:
        try:
            self._validate_matches_calculated()
            
            if len(self.image1.shape) == 2:
                img1_color = cv2.cvtColor(self.image1, cv2.COLOR_GRAY2BGR)
            else:
                img1_color = self.image1
                
            if len(self.image2.shape) == 2:
                img2_color = cv2.cvtColor(self.image2, cv2.COLOR_GRAY2BGR)
            else:
                img2_color = self.image2
            
            output_image = cv2.drawMatches(
                img1_color, self.keypoints1,
                img2_color, self.keypoints2,
                self.good_matches, output_image,
                flags=flags,
                matchColor=(0, 255, 0),  
                singlePointColor=(255, 0, 0),  
                matchesThickness=2
            )
            
            return output_image
            
        except Exception as e:
            raise RuntimeError(f"فشل في رسم المطابقات: {str(e)}")
    
    def get_match_statistics(self) -> Dict[str, Any]:
        try:
            self._validate_matches_calculated()
            
            if not self.good_matches:
                return {}
            
            distances = [m.distance for m in self.good_matches]
            
            statistics = {
                'total_matches': len(self.good_matches),
                'min_distance': min(distances),
                'max_distance': max(distances),
                'avg_distance': sum(distances) / len(distances),
                'matching_method': self.matching_method.value if self.matching_method else "Unknown",
                'keypoints_image1': len(self.keypoints1),
                'keypoints_image2': len(self.keypoints2)
            }
            
            return statistics
            
        except Exception as e:
            raise RuntimeError(f"فشل في حساب الإحصاءات: {str(e)}")
    
    def get_matching_results(self) -> Dict[str, Any]:
        try:
            return {
                'keypoints1': self.keypoints1,
                'keypoints2': self.keypoints2,
                'descriptors1': self.descriptors1,
                'descriptors2': self.descriptors2,
                'matches': self.matches,
                'good_matches': self.good_matches,
                'homography': self.homography,
                'statistics': self.get_match_statistics()
            }
        except Exception as e:
            raise RuntimeError(f"فشل في الحصول على النتائج: {str(e)}")


if __name__ == "__main__":
    try:
        img1 = cv2.imread('image1.jpg')
        img2 = cv2.imread('image2.jpg')
        
        if img1 is None or img2 is None:
            raise FileNotFoundError("لم يتم العثور على الصور")
        
        print("اختر طريقة المطابقة:")
        print("1. FLANN Based Matcher")
        print("2. Brute-Force Matching with SIFT Descriptor and Ratio Test")
        print("3. Brute-Force Matching with ORB Descriptor and Ratio Test")
        
        choice = input("أدخل رقم الطريقة (1-3): ").strip()
        
        if choice == "1":
            method = MatchingMethod.FLANN
        elif choice == "2":
            method = MatchingMethod.BF_SIFT_RATIO
        elif choice == "3":
            method = MatchingMethod.BF_ORB_RATIO
        else:
            raise ValueError("اختيار غير صالح")
        
        matcher = FeatureMatching(img1, img2)
        
        result_image = matcher \
            .detect_features(method) \
            .match_features(ratio_threshold=0.75) \
            .draw_matches()
        
        cv2.imshow('Feature Matches', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        stats = matcher.get_match_statistics()
        print("\nإحصاءات المطابقات:")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
        cv2.imwrite('feature_matches_result.jpg', result_image)
        print("تم حفظ الصورة الناتجة كـ 'feature_matches_result.jpg'")
            
    except Exception as e:
        print(f"حدث خطأ: {str(e)}")


