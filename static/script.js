// متغيرات عامة
let currentImage = null;
let originalImage = null;
let secondImage = null;

// عناصر DOM
const imageInput = document.getElementById('imageInput');
const secondImageInput = document.getElementById('secondImageInput');
const imageDisplay = document.getElementById('imageDisplay');
const loadingIndicator = document.getElementById('loadingIndicator');
const resultStats = document.getElementById('resultStats');

// إعداد الأحداث عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    setupTabs();
    setupDragAndDrop();
    setupParameterUpdates();
});

// إعداد مستمعات الأحداث
function setupEventListeners() {
    imageInput.addEventListener('change', handleImageUpload);
    secondImageInput.addEventListener('change', handleSecondImageUpload);
}

// إعداد التبويبات
function setupTabs() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // إزالة active من جميع التبويبات
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(tc => tc.classList.remove('active'));
            
            // إضافة active للتبويب المحدد
            tab.classList.add('active');
            const targetTab = tab.getAttribute('data-tab');
            document.getElementById(targetTab + '-tab').classList.add('active');
        });
    });
}

// إعداد السحب والإفلات
function setupDragAndDrop() {
    const uploadArea = document.querySelector('.upload-area');
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleImageFile(files[0]);
        }
    });
}

// إعداد تحديث المعاملات
function setupParameterUpdates() {
    // تحديث معاملات الميزات
    document.getElementById('featureType').addEventListener('change', updateFeatureParameters);
    
    // تحديث معاملات المرشحات
    document.getElementById('filterType').addEventListener('change', updateFilterParameters);
    
    // تحديث معاملات التحويل
    document.getElementById('transformType').addEventListener('change', updateTransformParameters);
}

// معالجة رفع الصورة
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        handleImageFile(file);
    }
}

// معالجة رفع الصورة الثانية
function handleSecondImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            secondImage = e.target.result;
            showMessage('تم تحميل الصورة الثانية بنجاح', 'success');
        };
        reader.readAsDataURL(file);
    }
}

// معالجة ملف الصورة
function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showMessage('يرجى اختيار ملف صورة صالح', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        currentImage = e.target.result;
        originalImage = e.target.result;
        displayImage(currentImage);
        showMessage('تم تحميل الصورة بنجاح', 'success');
    };
    reader.readAsDataURL(file);
}

// عرض الصورة
function displayImage(imageSrc) {
    imageDisplay.innerHTML = `<img src="${imageSrc}" alt="الصورة المحملة">`;
}

// عرض الرسائل
function showMessage(message, type = 'info') {
    const messageDiv = document.createElement('div');
    messageDiv.className = `${type}-message`;
    messageDiv.textContent = message;
    
    // إضافة الرسالة إلى أعلى الصفحة
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);
    
    // إزالة الرسالة بعد 5 ثوان
    setTimeout(() => {
        messageDiv.remove();
    }, 5000);
}

// عرض مؤشر التحميل
function showLoading() {
    loadingIndicator.classList.add('show');
}

// إخفاء مؤشر التحميل
function hideLoading() {
    loadingIndicator.classList.remove('show');
}

// تحديث معاملات الميزات
function updateFeatureParameters() {
    const featureType = document.getElementById('featureType').value;
    const paramsSection = document.getElementById('featureParams');
    
    let paramsHTML = '';
    
    switch(featureType) {
        case 'sift':
            paramsHTML = `
                <h4>معاملات SIFT</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">عدد الميزات</label>
                        <input type="number" class="form-input" id="siftFeatures" value="0" min="0">
                    </div>
                    <div class="form-group">
                        <label class="form-label">عتبة التباين</label>
                        <input type="number" class="form-input" id="contrastThreshold" value="0.04" step="0.01" min="0">
                    </div>
                </div>
            `;
            break;
        case 'orb':
            paramsHTML = `
                <h4>معاملات ORB</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">عدد الميزات</label>
                        <input type="number" class="form-input" id="orbFeatures" value="500" min="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">معامل التحجيم</label>
                        <input type="number" class="form-input" id="scaleFactor" value="1.2" step="0.1" min="1">
                    </div>
                </div>
            `;
            break;
        case 'fast_corners':
            paramsHTML = `
                <h4>معاملات FAST Corners</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">العتبة</label>
                        <input type="number" class="form-input" id="fastThreshold" value="10" min="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">قمع غير الأقصى</label>
                        <select class="form-select" id="nonmaxSuppression">
                            <option value="true">نعم</option>
                            <option value="false">لا</option>
                        </select>
                    </div>
                </div>
            `;
            break;
        case 'hog':
            paramsHTML = `
                <h4>معاملات HOG</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">عدد الصناديق</label>
                        <input type="number" class="form-input" id="hogBins" value="9" min="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">حجم الخلية</label>
                        <input type="number" class="form-input" id="cellSize" value="8" min="1">
                    </div>
                </div>
            `;
            break;
        case 'log_dog_blob':
            paramsHTML = `
                <h4>معاملات LoG/DoG Blob</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">العتبة الدنيا</label>
                        <input type="number" class="form-input" id="minThreshold" value="10" min="0">
                    </div>
                    <div class="form-group">
                        <label class="form-label">العتبة العليا</label>
                        <input type="number" class="form-input" id="maxThreshold" value="200" min="0">
                    </div>
                </div>
            `;
            break;
    }
    
    paramsSection.innerHTML = paramsHTML;
}

// تحديث معاملات المرشحات
function updateFilterParameters() {
    const filterType = document.getElementById('filterType').value;
    const paramsSection = document.getElementById('filterParams');
    
    let paramsHTML = '';
    
    switch(filterType) {
        case 'gaussian_blur':
            paramsHTML = `
                <h4>معاملات التمويه الغاوسي</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">حجم النواة</label>
                        <input type="number" class="form-input" id="kernelSize" value="5" min="1" step="2">
                    </div>
                    <div class="form-group">
                        <label class="form-label">سيجما</label>
                        <input type="number" class="form-input" id="sigma" value="0" min="0" step="0.1">
                    </div>
                </div>
            `;
            break;
        case 'canny':
            paramsHTML = `
                <h4>معاملات كاني</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">العتبة الأولى</label>
                        <input type="number" class="form-input" id="threshold1" value="100" min="0">
                    </div>
                    <div class="form-group">
                        <label class="form-label">العتبة الثانية</label>
                        <input type="number" class="form-input" id="threshold2" value="200" min="0">
                    </div>
                </div>
            `;
            break;
        case 'gamma_correction':
            paramsHTML = `
                <h4>معاملات تصحيح جاما</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">قيمة جاما</label>
                        <input type="number" class="form-input" id="gamma" value="1.0" step="0.1" min="0.1">
                    </div>
                </div>
            `;
            break;
        default:
            paramsHTML = `<h4>لا توجد معاملات إضافية لهذا المرشح</h4>`;
    }
    
    paramsSection.innerHTML = paramsHTML;
}

// تحديث معاملات التحويل
function updateTransformParameters() {
    const transformType = document.getElementById('transformType').value;
    const paramsSection = document.getElementById('transformParams');
    
    let paramsHTML = '';
    
    switch(transformType) {
        case 'rotate':
            paramsHTML = `
                <h4>معاملات الدوران</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">الزاوية (درجة)</label>
                        <input type="number" class="form-input" id="rotationAngle" value="45" step="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">معامل التحجيم</label>
                        <input type="number" class="form-input" id="rotationScale" value="1.0" step="0.1" min="0.1">
                    </div>
                </div>
            `;
            break;
        case 'translate':
            paramsHTML = `
                <h4>معاملات الإزاحة</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">الإزاحة الأفقية</label>
                        <input type="number" class="form-input" id="translateX" value="50" step="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">الإزاحة الرأسية</label>
                        <input type="number" class="form-input" id="translateY" value="50" step="1">
                    </div>
                </div>
            `;
            break;
        case 'scale':
            paramsHTML = `
                <h4>معاملات التحجيم</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">معامل التحجيم الأفقي</label>
                        <input type="number" class="form-input" id="scaleX" value="1.5" step="0.1" min="0.1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">معامل التحجيم الرأسي</label>
                        <input type="number" class="form-input" id="scaleY" value="1.5" step="0.1" min="0.1">
                    </div>
                </div>
            `;
            break;
        case 'flip':
            paramsHTML = `
                <h4>معاملات القلب</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">نوع القلب</label>
                        <select class="form-select" id="flipCode">
                            <option value="1">أفقي</option>
                            <option value="0">رأسي</option>
                            <option value="-1">أفقي ورأسي</option>
                        </select>
                    </div>
                </div>
            `;
            break;
        case 'crop':
            paramsHTML = `
                <h4>معاملات الاقتصاص</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">الإحداثي X</label>
                        <input type="number" class="form-input" id="cropX" value="0" min="0">
                    </div>
                    <div class="form-group">
                        <label class="form-label">الإحداثي Y</label>
                        <input type="number" class="form-input" id="cropY" value="0" min="0">
                    </div>
                    <div class="form-group">
                        <label class="form-label">العرض</label>
                        <input type="number" class="form-input" id="cropWidth" value="100" min="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">الارتفاع</label>
                        <input type="number" class="form-input" id="cropHeight" value="100" min="1">
                    </div>
                </div>
            `;
            break;
        case 'resize':
            paramsHTML = `
                <h4>معاملات تغيير الحجم</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">العرض الجديد</label>
                        <input type="number" class="form-input" id="newWidth" value="300" min="1">
                    </div>
                    <div class="form-group">
                        <label class="form-label">الارتفاع الجديد</label>
                        <input type="number" class="form-input" id="newHeight" value="300" min="1">
                    </div>
                </div>
            `;
            break;
        case 'adjust_color':
            paramsHTML = `
                <h4>معاملات ضبط الألوان</h4>
                <div class="parameter-group">
                    <div class="form-group">
                        <label class="form-label">القناة اللونية</label>
                        <select class="form-select" id="colorChannel">
                            <option value="ALL">جميع القنوات</option>
                            <option value="RED">الأحمر</option>
                            <option value="GREEN">الأخضر</option>
                            <option value="BLUE">الأزرق</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label">قيمة التعديل</label>
                        <input type="number" class="form-input" id="colorValue" value="50" min="-255" max="255">
                    </div>
                </div>
            `;
            break;
    }
    
    paramsSection.innerHTML = paramsHTML;
}

// استخراج الميزات
async function extractFeatures() {
    if (!currentImage) {
        showMessage('يرجى تحميل صورة أولاً', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const featureType = document.getElementById('featureType').value;
        const parameters = getFeatureParameters(featureType);
        
        const response = await fetch('/api/image/extract_features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage,
                feature_type: featureType,
                parameters: parameters
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentImage = result.result_image;
            displayImage(currentImage);
            updateResultStats(result);
            showMessage('تم استخراج الميزات بنجاح', 'success');
        } else {
            showMessage(result.error || 'حدث خطأ في استخراج الميزات', 'error');
        }
    } catch (error) {
        showMessage('خطأ في الاتصال بالخادم: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// تطبيق المرشح
async function applyFilter() {
    if (!currentImage) {
        showMessage('يرجى تحميل صورة أولاً', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const filterType = document.getElementById('filterType').value;
        const parameters = getFilterParameters(filterType);
        
        const response = await fetch('/api/image/apply_filter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage,
                filter_type: filterType,
                parameters: parameters
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentImage = result.result_image;
            displayImage(currentImage);
            showMessage('تم تطبيق المرشح بنجاح', 'success');
        } else {
            showMessage(result.error || 'حدث خطأ في تطبيق المرشح', 'error');
        }
    } catch (error) {
        showMessage('خطأ في الاتصال بالخادم: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// مطابقة الميزات
async function matchFeatures() {
    if (!currentImage || !secondImage) {
        showMessage('يرجى تحميل صورتين للمطابقة', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const matchingMethod = document.getElementById('matchingMethod').value;
        const ratioThreshold = parseFloat(document.getElementById('ratioThreshold').value);
        
        const response = await fetch('/api/image/match_features', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image1: currentImage,
                image2: secondImage,
                matching_method: matchingMethod,
                ratio_threshold: ratioThreshold
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentImage = result.result_image;
            displayImage(currentImage);
            updateResultStats(result);
            showMessage('تم إجراء مطابقة الميزات بنجاح', 'success');
        } else {
            showMessage(result.error || 'حدث خطأ في مطابقة الميزات', 'error');
        }
    } catch (error) {
        showMessage('خطأ في الاتصال بالخادم: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// تحويل الصورة
async function transformImage() {
    if (!currentImage) {
        showMessage('يرجى تحميل صورة أولاً', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const transformType = document.getElementById('transformType').value;
        const parameters = getTransformParameters(transformType);
        
        const response = await fetch('/api/image/transform_image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage,
                transform_type: transformType,
                parameters: parameters
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            currentImage = result.result_image;
            displayImage(currentImage);
            showMessage('تم تطبيق التحويل بنجاح', 'success');
        } else {
            showMessage(result.error || 'حدث خطأ في تحويل الصورة', 'error');
        }
    } catch (error) {
        showMessage('خطأ في الاتصال بالخادم: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// الحصول على معاملات الميزات
function getFeatureParameters(featureType) {
    const parameters = {};
    
    switch(featureType) {
        case 'sift':
            const siftFeatures = document.getElementById('siftFeatures');
            const contrastThreshold = document.getElementById('contrastThreshold');
            if (siftFeatures) parameters.n_features = parseInt(siftFeatures.value);
            if (contrastThreshold) parameters.contrast_threshold = parseFloat(contrastThreshold.value);
            break;
        case 'orb':
            const orbFeatures = document.getElementById('orbFeatures');
            const scaleFactor = document.getElementById('scaleFactor');
            if (orbFeatures) parameters.n_features = parseInt(orbFeatures.value);
            if (scaleFactor) parameters.scale_factor = parseFloat(scaleFactor.value);
            break;
        case 'fast_corners':
            const fastThreshold = document.getElementById('fastThreshold');
            const nonmaxSuppression = document.getElementById('nonmaxSuppression');
            if (fastThreshold) parameters.threshold = parseInt(fastThreshold.value);
            if (nonmaxSuppression) parameters.nonmax_suppression = nonmaxSuppression.value === 'true';
            break;
        case 'hog':
            const hogBins = document.getElementById('hogBins');
            const cellSize = document.getElementById('cellSize');
            if (hogBins) parameters.nbins = parseInt(hogBins.value);
            if (cellSize) {
                const size = parseInt(cellSize.value);
                parameters.cell_size = [size, size];
            }
            break;
        case 'log_dog_blob':
            const minThreshold = document.getElementById('minThreshold');
            const maxThreshold = document.getElementById('maxThreshold');
            if (minThreshold) parameters.min_threshold = parseFloat(minThreshold.value);
            if (maxThreshold) parameters.max_threshold = parseFloat(maxThreshold.value);
            break;
    }
    
    return parameters;
}

// الحصول على معاملات المرشحات
function getFilterParameters(filterType) {
    const parameters = {};
    
    switch(filterType) {
        case 'gaussian_blur':
            const kernelSize = document.getElementById('kernelSize');
            const sigma = document.getElementById('sigma');
            if (kernelSize) parameters.ksize = parseInt(kernelSize.value);
            if (sigma) parameters.sigma = parseFloat(sigma.value);
            break;
        case 'canny':
            const threshold1 = document.getElementById('threshold1');
            const threshold2 = document.getElementById('threshold2');
            if (threshold1) parameters.threshold1 = parseFloat(threshold1.value);
            if (threshold2) parameters.threshold2 = parseFloat(threshold2.value);
            break;
        case 'gamma_correction':
            const gamma = document.getElementById('gamma');
            if (gamma) parameters.gamma = parseFloat(gamma.value);
            break;
    }
    
    return parameters;
}

// الحصول على معاملات التحويل
function getTransformParameters(transformType) {
    const parameters = {};
    
    switch(transformType) {
        case 'rotate':
            const rotationAngle = document.getElementById('rotationAngle');
            const rotationScale = document.getElementById('rotationScale');
            if (rotationAngle) parameters.angle = parseFloat(rotationAngle.value);
            if (rotationScale) parameters.scale = parseFloat(rotationScale.value);
            break;
        case 'translate':
            const translateX = document.getElementById('translateX');
            const translateY = document.getElementById('translateY');
            if (translateX) parameters.tx = parseInt(translateX.value);
            if (translateY) parameters.ty = parseInt(translateY.value);
            break;
        case 'scale':
            const scaleX = document.getElementById('scaleX');
            const scaleY = document.getElementById('scaleY');
            if (scaleX) parameters.fx = parseFloat(scaleX.value);
            if (scaleY) parameters.fy = parseFloat(scaleY.value);
            break;
        case 'flip':
            const flipCode = document.getElementById('flipCode');
            if (flipCode) parameters.flip_code = parseInt(flipCode.value);
            break;
        case 'crop':
            const cropX = document.getElementById('cropX');
            const cropY = document.getElementById('cropY');
            const cropWidth = document.getElementById('cropWidth');
            const cropHeight = document.getElementById('cropHeight');
            if (cropX) parameters.x = parseInt(cropX.value);
            if (cropY) parameters.y = parseInt(cropY.value);
            if (cropWidth) parameters.width = parseInt(cropWidth.value);
            if (cropHeight) parameters.height = parseInt(cropHeight.value);
            break;
        case 'resize':
            const newWidth = document.getElementById('newWidth');
            const newHeight = document.getElementById('newHeight');
            if (newWidth) parameters.new_width = parseInt(newWidth.value);
            if (newHeight) parameters.new_height = parseInt(newHeight.value);
            break;
        case 'adjust_color':
            const colorChannel = document.getElementById('colorChannel');
            const colorValue = document.getElementById('colorValue');
            if (colorChannel) parameters.channel = colorChannel.value;
            if (colorValue) parameters.value = parseInt(colorValue.value);
            break;
    }
    
    return parameters;
}

// تحديث إحصاءات النتائج
function updateResultStats(result) {
    let statsHTML = '';
    
    if (result.features) {
        Object.keys(result.features).forEach(key => {
            statsHTML += `
                <div class="stat-card">
                    <div class="stat-value">${result.features[key]}</div>
                    <div class="stat-label">${key}</div>
                </div>
            `;
        });
    }
    
    if (result.statistics) {
        Object.keys(result.statistics).forEach(key => {
            let value = result.statistics[key];
            if (typeof value === 'number' && !Number.isInteger(value)) {
                value = value.toFixed(2);
            }
            statsHTML += `
                <div class="stat-card">
                    <div class="stat-value">${value}</div>
                    <div class="stat-label">${key}</div>
                </div>
            `;
        });
    }
    
    if (result.keypoints_count !== undefined) {
        statsHTML += `
            <div class="stat-card">
                <div class="stat-value">${result.keypoints_count}</div>
                <div class="stat-label">عدد النقاط المميزة</div>
            </div>
        `;
    }
    
    resultStats.innerHTML = statsHTML;
}

// إعادة تعيين الصورة
function resetImage() {
    if (originalImage) {
        currentImage = originalImage;
        displayImage(currentImage);
        resultStats.innerHTML = '';
        showMessage('تم إعادة تعيين الصورة إلى الحالة الأصلية', 'success');
    } else {
        showMessage('لا توجد صورة أصلية للعودة إليها', 'error');
    }
}

// تحميل النتيجة
function downloadResult() {
    if (!currentImage) {
        showMessage('لا توجد صورة للتحميل', 'error');
        return;
    }
    
    // إنشاء رابط تحميل
    const link = document.createElement('a');
    link.download = 'processed_image.png';
    link.href = currentImage;
    link.click();
    
    showMessage('تم تحميل الصورة بنجاح', 'success');
}

