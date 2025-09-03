from flask import Blueprint, request, jsonify, send_file
import cv2
import numpy as np
import base64
import io
import os
from PIL import Image
import tempfile
import traceback
from werkzeug.utils import secure_filename

from src.service.feature import AdvancedFeatureExtractor, FeatureType
from src.service.filters import AdvancedImageProcessor, FilterType
from src.service.match import FeatureMatching, MatchingMethod
from src.service.transformation import GeometricTransformation, GeometricTransformationType, ColorChannel

image_bp = Blueprint("image_processing", __name__)

UPLOAD_FOLDER = os.getcwd() + "/tmp/image_processing"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image):
    try:
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = Image.fromarray(image)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        print(f"Error encoding image: {str(e)}")
        return None

def decode_base64_to_image(base64_string):
    try:
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        image_array = np.array(image)
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    except Exception as e:
        print(f"Error decoding base64 image: {str(e)}")
        return None

@image_bp.route("/upload", methods=["POST"])
def upload_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "لم يتم العثور على ملف"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "لم يتم اختيار ملف"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            image = cv2.imread(filepath)
            if image is None:
                return jsonify({"error": "فشل في قراءة الصورة"}), 400
            
            base64_image = encode_image_to_base64(image)
            
            return jsonify({
                "success": True,
                "filename": filename,
                "image": base64_image,
                "shape": image.shape
            })
        
        return jsonify({"error": "نوع الملف غير مدعوم"}), 400
        
    except Exception as e:
        return jsonify({"error": f"خطأ في رفع الصورة: {str(e)}"}), 500

@image_bp.route("/extract_feature", methods=["POST"])
def extract_feature():
    try:
        data = request.get_json()
        
        if "image" not in data:
            return jsonify({"error": "الصورة مطلوبة"}), 400
        
        image = decode_base64_to_image(data["image"])
        if image is None:
            return jsonify({"error": "فشل في تحويل الصورة"}), 400
        
        feature_type = data.get("feature_type", "sift")
        parameters = data.get("parameters", {})
        
        extractor = AdvancedFeatureExtractor(image)
        
        result = extractor.extract_features(FeatureType(feature_type), **parameters)
        
        result_image_base64 = encode_image_to_base64(result.image)
        
        keypoint_data = []
        if result.keypoints:
            for kp in result.keypoints:
                keypoint_data.append({
                    "x": float(kp.pt[0]),
                    "y": float(kp.pt[1]),
                    "size": float(kp.size),
                    "angle": float(kp.angle),
                    "response": float(kp.response)
                })
        
        response_data = {
            "success": True,
            "result_image": result_image_base64,
            "features": result.features,
            "keypoint_data": keypoint_data[:10],  # Limit to first 10 for brevity
            "metadata": {
                "method": result.metadata.get("method"),
                "parameters": {k: str(v) for k, v in result.metadata.get("parameters", {}).items() 
                             if k not in ["self", "image"]}
            },
            "keypoint_count": len(result.keypoints) if result.keypoints else 0
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": f"خطأ في استخراج الميزات: {str(e)}"}), 500

@image_bp.route("/apply_filter", methods=["POST"])
def apply_filter():
    try:
        data = request.get_json()
        
        if "image" not in data:
            return jsonify({"error": "الصورة مطلوبة"}), 400
        
        image = decode_base64_to_image(data["image"])
        if image is None:
            return jsonify({"error": "فشل في تحويل الصورة"}), 400
        
        filter_type = data.get("filter_type", "gaussian_blur")
        parameters = data.get("parameters", {})
        
        processor = AdvancedImageProcessor(image)
        
        result_image = processor.apply_filter(FilterType(filter_type), **parameters)
        
        result_image_base64 = encode_image_to_base64(result_image)
        
        return jsonify({
            "success": True,
            "result_image": result_image_base64,
            "filter_type": filter_type,
            "parameters": parameters
        })
        
    except Exception as e:
        return jsonify({"error": f"خطأ في تطبيق المرشح: {str(e)}"}), 500

@image_bp.route("/match_feature", methods=["POST"])
def match_feature():
    try:
        data = request.get_json()
        
        if "image1" not in data or "image2" not in data:
            return jsonify({"error": "الصورتان مطلوبتان"}), 400
        
        image1 = decode_base64_to_image(data["image1"])
        image2 = decode_base64_to_image(data["image2"])
        
        if image1 is None or image2 is None:
            return jsonify({"error": "فشل في تحويل الصور"}), 400
        
        matching_method = data.get("matching_method", "BF_SIFT_RATIO")
        ratio_threshold = data.get("ratio_threshold", 0.75)
        max_distance = data.get("max_distance", 100.0)
        
        matcher = FeatureMatching(image1, image2)
        
        matcher.detect_features(MatchingMethod(matching_method))
        matcher.match_features(ratio_threshold=ratio_threshold, max_distance=max_distance)
        
        result_image = matcher.draw_matches()
        
        statistics = matcher.get_match_statistics()
        
        result_image_base64 = encode_image_to_base64(result_image)
        
        return jsonify({
            "success": True,
            "result_image": result_image_base64,
            "statistics": statistics,
            "matching_method": matching_method
        })
        
    except Exception as e:
        return jsonify({"error": f"خطأ في مطابقة الميزات: {str(e)}"}), 500

@image_bp.route("/transform_image", methods=["POST"])
def transform_image():
    try:
        data = request.get_json()
        
        if "image" not in data:
            return jsonify({"error": "الصورة مطلوبة"}), 400
        
        image = decode_base64_to_image(data["image"])
        if image is None:
            return jsonify({"error": "فشل في تحويل الصورة"}), 400
        
        transform_type = data.get("transform_type", "rotate")
        parameters = data.get("parameters", {})
        
        transformer = GeometricTransformation(image)
        
        if transform_type == "rotate":
            angle = parameters.get("angle", 45)
            center = parameters.get("center")
            scale = parameters.get("scale", 1.0)
            transformer.rotate(angle, center, scale)
        elif transform_type == "translate":
            tx = parameters.get("tx", 50)
            ty = parameters.get("ty", 50)
            transformer.translate(tx, ty)
        elif transform_type == "scale":
            fx = parameters.get("fx", 1.5)
            fy = parameters.get("fy", fx)
            transformer.scale(fx, fy)
        elif transform_type == "flip":
            flip_code = parameters.get("flip_code", 1)
            transformer.flip(flip_code)
        elif transform_type == "crop":
            x = parameters.get("x", 0)
            y = parameters.get("y", 0)
            width = parameters.get("width", 100)
            height = parameters.get("height", 100)
            transformer.crop(x, y, width, height)
        elif transform_type == "resize":
            new_width = parameters.get("new_width", 300)
            new_height = parameters.get("new_height", 300)
            transformer.resize(new_width, new_height)
        elif transform_type == "adjust_color":
            channel = parameters.get("channel", "ALL")
            value = parameters.get("value", 50)
            transformer.adjust_color_channel(ColorChannel[channel], value)
        else:
            return jsonify({"error": f"نوع التحويل غير مدعوم: {transform_type}"}), 400
        
        result_image = transformer.get_current_image()
        
        result_image_base64 = encode_image_to_base64(result_image)
        
        return jsonify({
            "success": True,
            "result_image": result_image_base64,
            "transform_type": transform_type,
            "parameters": parameters
        })
        
    except Exception as e:
        return jsonify({"error": f"خطأ في تحويل الصورة: {str(e)}"}), 500

@image_bp.route("/batch_process", methods=["POST"])
def batch_process():
    try:
        data = request.get_json()
        
        if "images" not in data or "operations" not in data:
            return jsonify({"error": "الصور والعمليات مطلوبة"}), 400
        
        image_data_list = data["images"]
        operations = data["operations"]
        
        results = []
        
        for i, image_data in enumerate(image_data_list):
            try:
                image = decode_base64_to_image(image_data)
                if image is None:
                    results.append({"error": f"فشل في تحويل الصورة {i+1}"})
                    continue
                
                current_image = image
                
                for operation in operations:
                    op_type = operation.get("type")
                    op_params = operation.get("parameters", {})
                    
                    if op_type == "filter":
                        processor = AdvancedImageProcessor(current_image)
                        current_image = processor.apply_filter(
                            FilterType(op_params.get("filter_type", "gaussian_blur")),
                            **op_params.get("filter_params", {})
                        )
                    elif op_type == "transform":
                        transformer = GeometricTransformation(current_image)
                        transform_type = op_params.get("transform_type", "rotate")
                        
                        if transform_type == "rotate":
                            transformer.rotate(
                                op_params.get("angle", 45),
                                op_params.get("center"),
                                op_params.get("scale", 1.0)
                            )
                        elif transform_type == "scale":
                            transformer.scale(
                                op_params.get("fx", 1.5),
                                op_params.get("fy")
                            )
                        elif transform_type == "translate":
                            transformer.translate(
                                op_params.get("tx", 50),
                                op_params.get("ty", 50)
                            )
                        elif transform_type == "flip":
                            transformer.flip(
                                op_params.get("flip_code", 1)
                            )
                        elif transform_type == "crop":
                            transformer.crop(
                                op_params.get("x", 0),
                                op_params.get("y", 0),
                                op_params.get("width", 100),
                                op_params.get("height", 100)
                            )
                        elif transform_type == "resize":
                            transformer.resize(
                                op_params.get("new_width", 300),
                                op_params.get("new_height", 300)
                            )
                        elif transform_type == "adjust_color":
                            transformer.adjust_color_channel(
                                ColorChannel[op_params.get("channel", "ALL")],
                                op_params.get("value", 50)
                            )
                        else:
                            raise ValueError(f"نوع التحويل غير مدعوم في المعالجة الدفعية: {transform_type}")
                        
                        current_image = transformer.get_current_image()
                
                result_base64 = encode_image_to_base64(current_image)
                results.append({
                    "success": True,
                    "result_image": result_base64,
                    "index": i
                })
                
            except Exception as e:
                results.append({
                    "error": f"خطأ في معالجة الصورة {i+1}: {str(e)}",
                    "index": i
                })
        
        return jsonify({
            "success": True,
            "results": results,
            "total_processed": len(results)
        })
        
    except Exception as e:
        return jsonify({"error": f"خطأ في المعالجة الدفعية: {str(e)}"}), 500

@image_bp.route("/get_available_options", methods=["GET"])
def get_available_options():
    try:
        options = {
            "feature_types": [ft.value for ft in FeatureType],
            "filter_types": [ft.value for ft in FilterType],
            "matching_methods": [mm.value for mm in MatchingMethod],
            "transform_types": [
                "rotate", "translate", "scale", "flip", 
                "crop", "resize", "adjust_color"
            ],
            "color_channels": [cc.name for cc in ColorChannel]
        }
        
        return jsonify({
            "success": True,
            "options": options
        })
        
    except Exception as e:
        return jsonify({"error": f"خطأ في الحصول على الخيارات: {str(e)}"}), 500

@image_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "success": True,
        "message": "خدمة معالجة الصور تعمل بشكل طبيعي",
        "version": "1.0.0"
    })


