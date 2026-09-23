import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# إعدادات صفحة الويب
st.set_page_config(page_title="نظام تشخيص التهاب الرئة", page_icon="🩺", layout="centered")

st.title("🩺 نظام فحص وتشخيص أشعة الصدر الذكي")
st.write("قم برفع صورة أشعة الصدر (X-ray) وسيقوم النظام بفحصها والتأكد من صحتها ثم إعطائك تقريراً تشخيصياً مفصلاً.")

# تحميل النموذج المحفوظ مسبقاً
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('chest_xray_model.keras')

with st.spinner("جاري تحميل النموذج الذكي..."):
    model = load_model()

# أداة رفع الصورة من جهاز المستخدم
uploaded_file = st.file_uploader("اختر صورة أشعة الصدر (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # فتح وعرض الصورة المرفوعة
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="الصورة المُفحوصة", use_container_width=True)
    
    # تحويل الصورة إلى مصفوفة لتطبيق حراس الأمان
    img_array_raw = np.array(image)
    img_decoded = tf.constant(img_array_raw)
    
    with st.spinner("جاري فحص وتدقيق الصورة بواسطة الحراس الأمنيين..."):
        # --- 1. حارس الألوان الأمني ---
        img_float = tf.cast(img_decoded, tf.float32)
        r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
        max_color_diff = tf.reduce_max(tf.abs(r - g)) + tf.reduce_max(tf.abs(g - b))

        if max_color_diff > 25.0:
            st.error("🛑 تنبيه أمني: هذه الصورة ملونة وليست أشعة صدر رمادية نقية! تم رفض التشخيص.")
        else:
            # --- 2. حارس البنية والملمس الدقيق ---
            gray_img = tf.image.rgb_to_grayscale(img_decoded) if img_decoded.shape[-1] == 3 else img_decoded
            gray_float = tf.cast(gray_img, tf.float32)
            
            std_dev = tf.math.reduce_std(gray_float).numpy()
            gray_4d = tf.expand_dims(gray_float, axis=0)
            dy, dx = tf.image.image_gradients(gray_4d)
            edge_mag = tf.sqrt(dx**2 + dy**2)
            mean_edge = tf.reduce_mean(edge_mag).numpy()

            if mean_edge > 7.5:
                st.error("❌ رفض أمني: الصورة ليست أشعة صدر طبية خالصة! (تم اكتشاف عناصر أو حواف غير طبية).")
            else:
                # تجهيز الصورة وتمريرها للنموذج الطبي
                img_resized = tf.image.resize(img_decoded, [224, 224])
                img_array = tf.expand_dims(img_resized, axis=0)

                predictions = model.predict(img_array, verbose=0)[0]
                prob_normal = predictions[0] * 100
                prob_abnormal = predictions[1] * 100

                # عرض النتائج في واجهة المستخدم
                st.markdown("---")
                st.subheader("📊 التقرير التشخيصي المفصل")
                
                col1, col2 = st.columns(2)
                col1.metric(label="نسبة سليم (Normal)", value=f"{prob_normal:.2f}%")
                col2.metric(label="نسبة مرض (Abnormal)", value=f"{prob_abnormal:.2f}%")
                
                st.markdown("---")
                if prob_abnormal > prob_normal:
                    st.error("🩺 **التشخيص النهائي: Abnormal (التهاب رئوي)**")
                    if prob_abnormal >= 85:
                        st.info("📋 **التوصيف الإشعاعي:** النموذج يرصد مؤشرات واضحة وقوية جداً وجود تعتيمات أو كثافة غير طبيعية في أنسجة الرئة.")
                    else:
                        st.info("📋 **التوصيف الإشعاعي:** توجد مؤشرات مرجحة للمرض، ويُفضل عرض الصورة على طبيب مختص للتأكد.")
                else:
                    st.success("🩺 **التشخيص النهائي: Normal (سليم)**")
                    if prob_normal >= 85:
                        st.info("📋 **التوصيف الإشعاعي:** الرئتان تبدوان نقيتين، مع وضوح الأنسجة وعدم وجود علامات تدل على ترشيح أو التهاب رئوي.")
                    else:
                        st.info("📋 **التوصيف الإشعاعي:** الصورة تميل للحالة الطبيعية مع وجود بعض التباين العادي.")