import cv2
import numpy as np
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional, Union

class GeometricTransformationType(Enum):
    TRANSLATION = "translation"
    ROTATION = "rotation"
    SCALING = "scaling"
    SHEAR = "shear"
    AFFINE = "affine"
    PERSPECTIVE = "perspective"
    FLIP = "flip"
    CROP = "crop"
    RESIZE = "resize"
    WARP_POLAR = "warp_polar"
    COLOR_ADJUSTMENT = "color_adjustment"

class ColorChannel(Enum):
    RED = 2
    GREEN = 1
    BLUE = 0
    ALL = -1

class GeometricTransformation:
    def __init__(self, image: np.ndarray):
        if image is None:
            raise ValueError("الصورة المدخلة لا يمكن أن تكون None")
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"نوع الصورة يجب أن يكون numpy.ndarray، لكن تم إدخال: {type(image)}")
        
        if image.size == 0:
            raise ValueError("الصورة المدخلة فارغة")
        
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.transformation_history = []
        
    def reset(self):
        self.current_image = self.original_image.copy()
        self.transformation_history.clear()
        
    def get_current_image(self):
        return self.current_image.copy()
    
    def get_original_image(self):
        return self.original_image.copy()
    
    def get_history(self):
        return self.transformation_history.copy()
    
    def _validate_point(self, point, expected_count, name):
        if point is None:
            raise ValueError(f"النقاط {name} لا يمكن أن تكون None")
        
        if not isinstance(point, np.ndarray):
            raise TypeError(f"نقاط {name} يجب أن تكون numpy.ndarray")
        
        if point.shape[0] != expected_count:
            raise ValueError(f"عدد نقاط {name} يجب أن يكون {expected_count}")
        
        if point.shape[1] != 2:
            raise ValueError(f"كل نقطة في {name} يجب أن تحتوي على إحداثيين (x, y)")
    
    def _validate_parameter(self, **kwargs):
        for param_name, param_value in kwargs.items():
            if param_value is None:
                raise ValueError(f"معامل {param_name} لا يمكن أن يكون None")

    def adjust_color_channel(self, channel: ColorChannel, value: int):
        try:
            if not isinstance(channel, ColorChannel):
                raise ValueError("القناة اللونية يجب أن تكون من نوع ColorChannel")
            
            if value < -255 or value > 255:
                raise ValueError("قيمة الضبط يجب أن تكون بين -255 و 255")
            
            if channel == ColorChannel.ALL:
                adjusted_image = self.current_image.astype(np.int16) + value
                adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
            else:
                channel_idx = channel.value
                adjusted_image = self.current_image.copy()
                adjusted_image[:, :, channel_idx] = np.clip(
                    adjusted_image[:, :, channel_idx].astype(np.int16) + value, 0, 255
                ).astype(np.uint8)
            
            self.current_image = adjusted_image
            self.transformation_history.append({
                'type': GeometricTransformationType.COLOR_ADJUSTMENT,
                'parameters': {'channel': channel, 'value': value}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في ضبط القناة اللونية: {str(e)}")

    def set_color_channel(self, channel: ColorChannel, value: int):
        try:
            if not isinstance(channel, ColorChannel):
                raise ValueError("القناة اللونية يجب أن تكون من نوع ColorChannel")
            
            if value < 0 or value > 255:
                raise ValueError("قيمة القناة يجب أن تكون بين 0 و 255")
            
            if channel == ColorChannel.ALL:
                adjusted_image = np.full_like(self.current_image, value)
            else:
                channel_idx = channel.value
                adjusted_image = self.current_image.copy()
                adjusted_image[:, :, channel_idx] = value
            
            self.current_image = adjusted_image
            self.transformation_history.append({
                'type': GeometricTransformationType.COLOR_ADJUSTMENT,
                'parameters': {'channel': channel, 'set_value': value}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تعيين القناة اللونية: {str(e)}")

    def multiply_color_channel(self, channel: ColorChannel, factor: float):
        try:
            if not isinstance(channel, ColorChannel):
                raise ValueError("القناة اللونية يجب أن تكون من نوع ColorChannel")
            
            if factor <= 0:
                raise ValueError("معامل الضرب يجب أن يكون قيمة موجبة")
            
            if channel == ColorChannel.ALL:
                adjusted_image = (self.current_image.astype(np.float32) * factor)
                adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)
            else:
                channel_idx = channel.value
                adjusted_image = self.current_image.copy()
                adjusted_image[:, :, channel_idx] = np.clip(
                    adjusted_image[:, :, channel_idx].astype(np.float32) * factor, 0, 255
                ).astype(np.uint8)
            
            self.current_image = adjusted_image
            self.transformation_history.append({
                'type': GeometricTransformationType.COLOR_ADJUSTMENT,
                'parameters': {'channel': channel, 'multiply_factor': factor}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في ضرب القناة اللونية: {str(e)}")

    def translate(self, tx, ty, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
        try:
            self._validate_parameter(tx=tx, ty=ty)
            
            rows, cols = self.current_image.shape[:2]
            translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            
            transformed_image = cv2.warpAffine(
                self.current_image, translation_matrix, (cols, rows),
                borderMode=border_mode, borderValue=border_value
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.TRANSLATION,
                'parameters': {'tx': tx, 'ty': ty, 'border_mode': border_mode, 'border_value': border_value},
                'matrix': translation_matrix
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تحويل الإزاحة: {str(e)}")
    
    def rotate(self, angle, center=None, scale=1.0, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
        try:
            self._validate_parameter(angle=angle, scale=scale)
            
            rows, cols = self.current_image.shape[:2]
            
            if center is None:
                center = (cols / 2, rows / 2)
            
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            
            cos_angle = np.abs(rotation_matrix[0, 0])
            sin_angle = np.abs(rotation_matrix[0, 1])
            new_cols = int((rows * sin_angle) + (cols * cos_angle))
            new_rows = int((rows * cos_angle) + (cols * sin_angle))
            
            rotation_matrix[0, 2] += (new_cols / 2) - center[0]
            rotation_matrix[1, 2] += (new_rows / 2) - center[1]
            
            transformed_image = cv2.warpAffine(
                self.current_image, rotation_matrix, (new_cols, new_rows),
                borderMode=border_mode, borderValue=border_value
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.ROTATION,
                'parameters': {'angle': angle, 'center': center, 'scale': scale, 
                              'border_mode': border_mode, 'border_value': border_value},
                'matrix': rotation_matrix
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تدوير الصورة: {str(e)}")
    
    def scale(self, fx, fy=None, interpolation=cv2.INTER_LINEAR):
        try:
            self._validate_parameter(fx=fx)
            
            if fy is None:
                fy = fx
            
            if fx <= 0 or fy <= 0:
                raise ValueError("عوامل التحجيم يجب أن تكون قيم موجبة")
            
            rows, cols = self.current_image.shape[:2]
            new_cols = int(cols * fx)
            new_rows = int(rows * fy)
            
            transformed_image = cv2.resize(
                self.current_image, (new_cols, new_rows), 
                interpolation=interpolation
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.SCALING,
                'parameters': {'fx': fx, 'fy': fy, 'interpolation': interpolation}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تحجيم الصورة: {str(e)}")
    
    def affine_transform(self, src_points, dst_points, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
        try:
            self._validate_point(src_points, 3, "المصدر")
            self._validate_point(dst_points, 3, "الوجهة")
            
            rows, cols = self.current_image.shape[:2]
            affine_matrix = cv2.getAffineTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
            
            transformed_image = cv2.warpAffine(
                self.current_image, affine_matrix, (cols, rows),
                borderMode=border_mode, borderValue=border_value
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.AFFINE,
                'parameters': {'src_points': src_points, 'dst_points': dst_points,
                              'border_mode': border_mode, 'border_value': border_value},
                'matrix': affine_matrix
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في التحويل الأفيني: {str(e)}")
    
    def perspective_transform(self, src_points, dst_points, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
        try:
            self._validate_point(src_points, 4, "المصدر")
            self._validate_point(dst_points, 4, "الوجهة")
            
            rows, cols = self.current_image.shape[:2]
            perspective_matrix = cv2.getPerspectiveTransform(
                src_points.astype(np.float32), dst_points.astype(np.float32)
            )
            
            transformed_image = cv2.warpPerspective(
                self.current_image, perspective_matrix, (cols, rows),
                borderMode=border_mode, borderValue=border_value
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.PERSPECTIVE,
                'parameters': {'src_points': src_points, 'dst_points': dst_points,
                              'border_mode': border_mode, 'border_value': border_value},
                'matrix': perspective_matrix
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تحويل المنظور: {str(e)}")
    
    def flip(self, flip_code):
        try:
            if flip_code not in [0, 1, -1]:
                raise ValueError("كود القلب يجب أن يكون 0، 1، أو -1")
            
            transformed_image = cv2.flip(self.current_image, flip_code)
            self.current_image = transformed_image
            
            self.transformation_history.append({
                'type': GeometricTransformationType.FLIP,
                'parameters': {'flip_code': flip_code}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في قلب الصورة: {str(e)}")
    
    def crop(self, x, y, width, height):
        try:
            self._validate_parameter(x=x, y=y, width=width, height=height)
            
            rows, cols = self.current_image.shape[:2]
            
            if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                x + width > cols or y + height > rows):
                raise ValueError("إحداثيات الاقتصاص خارج نطاق الصورة")
            
            cropped_image = self.current_image[y:y+height, x:x+width]
            self.current_image = cropped_image
            
            self.transformation_history.append({
                'type': GeometricTransformationType.CROP,
                'parameters': {'x': x, 'y': y, 'width': width, 'height': height}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في اقتصاص الصورة: {str(e)}")
    
    def resize(self, new_width, new_height, interpolation=cv2.INTER_LINEAR):
        try:
            self._validate_parameter(new_width=new_width, new_height=new_height)
            
            if new_width <= 0 or new_height <= 0:
                raise ValueError("الأبعاد الجديدة يجب أن تكون قيم موجبة")
            
            transformed_image = cv2.resize(
                self.current_image, (new_width, new_height), 
                interpolation=interpolation
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.RESIZE,
                'parameters': {'new_width': new_width, 'new_height': new_height, 
                              'interpolation': interpolation}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تغيير حجم الصورة: {str(e)}")
    
    def warp_polar(self, center, max_radius, flags, output_size=(0, 0)):
        try:
            self._validate_parameter(center=center, max_radius=max_radius, flags=flags)
            
            if max_radius <= 0:
                raise ValueError("نصف القطر يجب أن يكون قيمة موجبة")
            
            transformed_image = cv2.warpPolar(
                self.current_image, output_size, center, max_radius, flags
            )
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': GeometricTransformationType.WARP_POLAR,
                'parameters': {'center': center, 'max_radius': max_radius, 
                              'flags': flags, 'output_size': output_size}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في التحويل القطبي: {str(e)}")
    
    def apply_custom_transformation(self, transformation_matrix, output_size=None, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
        try:
            if transformation_matrix is None:
                raise ValueError("مصفوفة التحويل لا يمكن أن تكون None")
            
            if not isinstance(transformation_matrix, np.ndarray):
                raise TypeError("مصفوفة التحويل يجب أن تكون numpy.ndarray")
            
            rows, cols = self.current_image.shape[:2]
            
            if output_size is None:
                output_size = (cols, rows)
            
            if transformation_matrix.shape == (2, 3):
                transformed_image = cv2.warpAffine(
                    self.current_image, transformation_matrix, output_size,
                    borderMode=border_mode, borderValue=border_value
                )
            elif transformation_matrix.shape == (3, 3):
                transformed_image = cv2.warpPerspective(
                    self.current_image, transformation_matrix, output_size,
                    borderMode=border_mode, borderValue=border_value
                )
            else:
                raise ValueError("مصفوفة التحويل يجب أن تكون إما 2x3 أو 3x3")
            
            self.current_image = transformed_image
            self.transformation_history.append({
                'type': 'custom',
                'parameters': {'transformation_matrix': transformation_matrix,
                              'output_size': output_size, 'border_mode': border_mode,
                              'border_value': border_value}
            })
            
            return self
            
        except Exception as e:
            raise RuntimeError(f"فشل في تطبيق التحويل المخصص: {str(e)}")

    def apply_transformation_chain(self, transformation_chain: List[Dict[str, Any]]) -> np.ndarray:
        try:
            for transform_config in transformation_chain:
                transform_type = transform_config.get("transform_type")
                params = transform_config.get("parameters", {})
                
                if transform_type == GeometricTransformationType.TRANSLATION.value:
                    self.translate(**params)
                elif transform_type == GeometricTransformationType.ROTATION.value:
                    self.rotate(**params)
                elif transform_type == GeometricTransformationType.SCALING.value:
                    self.scale(**params)
                elif transform_type == GeometricTransformationType.AFFINE.value:
                    self.affine_transform(**params)
                elif transform_type == GeometricTransformationType.PERSPECTIVE.value:
                    self.perspective_transform(**params)
                elif transform_type == GeometricTransformationType.FLIP.value:
                    self.flip(**params)
                elif transform_type == GeometricTransformationType.CROP.value:
                    self.crop(**params)
                elif transform_type == GeometricTransformationType.RESIZE.value:
                    self.resize(**params)
                elif transform_type == GeometricTransformationType.WARP_POLAR.value:
                    self.warp_polar(**params)
                elif transform_type == GeometricTransformationType.COLOR_ADJUSTMENT.value:
                    if "value" in params:
                        self.adjust_color_channel(**params)
                    elif "set_value" in params:
                        self.set_color_channel(**params)
                    elif "multiply_factor" in params:
                        self.multiply_color_channel(**params)
                else:
                    raise ValueError(f"نوع التحويل غير مدعوم في السلسلة: {transform_type}")
            
            return self.current_image
        except Exception as e:
            raise RuntimeError(f"فشل في تطبيق سلسلة التحويلات: {str(e)}")

    def save_current_image(self, filepath: str) -> bool:
        try:
            success = cv2.imwrite(filepath, self.current_image)
            if not success:
                raise IOError(f"فشل في حفظ الصورة إلى {filepath}")
            return success
        except Exception as e:
            raise RuntimeError(f"خطأ في حفظ الصورة: {str(e)}")

    def batch_transform(self, images: List[np.ndarray], transformation_chain: List[Dict[str, Any]]) -> List[np.ndarray]:
        results = []
        for img in images:
            try:
                transformer = GeometricTransformation(img)
                transformed_img = transformer.apply_transformation_chain(transformation_chain)
                results.append(transformed_img)
            except Exception as e:
                raise RuntimeError(f"فشل في معالجة الصورة في الدفعة: {str(e)}")
        return results

if __name__ == "__main__":
    try:
        # مثال على الاستخدام
        image_path = "example.jpg"  # تأكد من وجود هذه الصورة في نفس المجلد
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError(f"لم يتم العثور على الصورة في المسار: {image_path}")

        transformer = GeometricTransformation(img)

        # تطبيق سلسلة من التحويلات
        transformed_image = transformer \
            .translate(tx=50, ty=30) \
            .rotate(angle=45) \
            .scale(fx=0.5, fy=0.5) \
            .get_current_image()

        cv2.imshow("Transformed Image", transformed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # حفظ الصورة المحولة
        transformer.save_current_image("transformed_example.jpg")
        print("تم حفظ الصورة المحولة كـ transformed_example.jpg")

        # مثال على سلسلة تحويلات معقدة
        complex_chain = [
            {"transform_type": "crop", "parameters": {"x": 10, "y": 10, "width": 100, "height": 100}},
            {"transform_type": "resize", "parameters": {"new_width": 200, "new_height": 200}},
            {"transform_type": "color_adjustment", "parameters": {"channel": ColorChannel.RED, "value": 50}}
        ]

        transformer_complex = GeometricTransformation(img)
        transformed_complex_image = transformer_complex.apply_transformation_chain(complex_chain)

        cv2.imshow("Complex Transformed Image", transformed_complex_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"حدث خطأ: {str(e)}")


