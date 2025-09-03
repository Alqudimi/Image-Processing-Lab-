# مشروع معالجة الصور

هذا المشروع يوفر مجموعة من الأدوات لمعالجة الصور واستخراج الميزات منها.



## التثبيت

للبدء باستخدام هذا المشروع، اتبع الخطوات التالية:

1.  **استنساخ المستودع:**
    ```bash
    git clone https://github.com/your-username/image-processing-project.git
    cd image-processing-project
    ```

2.  **إنشاء بيئة افتراضية (موصى به):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # لنظامي التشغيل Linux/macOS
    # venv\Scripts\activate  # لنظام التشغيل Windows
    ```

3.  **تثبيت التبعيات:**
    ```bash
    pip install -r requirements.txt
    ```
    (ملاحظة: ستحتاج إلى إنشاء ملف `requirements.txt` يحتوي على جميع تبعيات المشروع مثل `opencv-python`, `numpy`, `fastapi`, `uvicorn`، إلخ.)



## الاستخدام

يمكن استخدام هذا المشروع لمعالجة الصور واستخراج الميزات المختلفة. فيما يلي بعض الأمثلة على كيفية استخدام المكونات الرئيسية:

### استخراج الميزات

يحتوي المشروع على وحدة `feature.py` التي توفر فئات لاستخراج الميزات مثل FAST Corners، HOG، LoG/DoG Blob، ORB، و SIFT.

```python
import cv2
from src.service.feature import AdvancedFeatureExtractor, FeatureType

# تحميل صورة
image = cv2.imread('path/to/your/image.jpg')

# إنشاء كائن AdvancedFeatureExtractor
extractor = AdvancedFeatureExtractor(image)

# استخراج ميزات FAST Corners
fast_result = extractor.extract_features(FeatureType.FAST_CORNERS, threshold=20)
print(f"عدد نقاط FAST Corners: {fast_result.features['keypoints_count']}")

# استخراج ميزات HOG
hog_result = extractor.extract_features(FeatureType.HOG)
print(f"طول متجه ميزات HOG: {hog_result.features['feature_vector_length']}")

# استخراج ميزات ORB
orb_result = extractor.extract_features(FeatureType.ORB, n_features=1000)
print(f"عدد نقاط ORB: {orb_result.features['keypoints_count']}")

# مطابقة الميزات (مثال)
# تحتاج إلى صورتين وميزات مستخرجة للمطابقة
# matches = extractor.match_features(orb_result.descriptors, another_orb_result.descriptors)
```

### المرشحات والتحويلات

يمكنك استخدام وحدتي `filters.py` و `transformation.py` لتطبيق مرشحات وتحويلات مختلفة على الصور.

```python
from src.service.filters import ImageFilter
from src.service.transformation import ImageTransformer

# تحميل صورة
image = cv2.imread('path/to/your/image.jpg')

# تطبيق مرشح التنعيم (Gaussian Blur)
filtered_image = ImageFilter.apply_gaussian_blur(image, kernel_size=(5, 5), sigma_x=0)

# تطبيق تحويل تغيير الحجم (Resize)
transformed_image = ImageTransformer.resize_image(image, width=200, height=200)
```

### تشغيل الخادم (إذا كان المشروع يتضمن واجهة برمجة تطبيقات)

إذا كان المشروع يتضمن واجهة برمجة تطبيقات (API) باستخدام FastAPI، يمكنك تشغيل الخادم باستخدام Uvicorn:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

ثم يمكنك الوصول إلى واجهة برمجة التطبيقات عبر `http://localhost:8000`.



## هيكل المشروع

يتكون المشروع من الهيكل التالي:

```
src/
├── __init__.py
├── main.py
├── routes/
│   ├── __init__.py
│   └── image_processing.py
├── service/
│   ├── __init__.py
│   ├── feature.py
│   ├── filters.py
│   ├── match.py
│   └── transformation.py
└── static/
    ├── favicon.ico
    ├── index.html
    ├── script.js
    └── tuition.html
```

-   `src/main.py`: نقطة الدخول الرئيسية للتطبيق.
-   `src/routes/`: يحتوي على تعريفات المسارات (endpoints) لواجهة برمجة التطبيقات.
-   `src/service/`: يحتوي على منطق العمل الأساسي والوظائف المتعلقة بمعالجة الصور واستخراج الميزات.
-   `src/static/`: يحتوي على الملفات الثابتة مثل ملفات HTML و CSS و JavaScript.



## المساهمة

نرحب بالمساهمات في هذا المشروع! إذا كنت ترغب في المساهمة، يرجى اتباع الإرشادات التالية:

1.  **تفرع (Fork) المستودع:** ابدأ بتفرع المستودع إلى حسابك على GitHub.
2.  **إنشاء فرع جديد:** أنشئ فرعًا جديدًا للميزة أو إصلاح الأخطاء التي تعمل عليها:
    ```bash
    git checkout -b feature/your-feature-name
    ```
3.  **إجراء التغييرات:** قم بإجراء التغييرات اللازمة واختبرها جيدًا.
4.  **الالتزام بالتغييرات:** التزم بتغييراتك برسالة التزام واضحة وموجزة:
    ```bash
    git commit -m "feat: Add new image filter"
    ```
5.  **دفع الفرع:** ادفع الفرع الجديد إلى مستودعك المتفرع:
    ```bash
    git push origin feature/your-feature-name
    ```
6.  **إنشاء طلب سحب (Pull Request):** افتح طلب سحب من فرعك إلى الفرع الرئيسي للمستودع الأصلي. يرجى وصف التغييرات التي أجريتها بوضوح وتقديم أي معلومات إضافية قد تكون مفيدة للمراجعة.

