# app.py
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import time
import numpy as np

# ===========================
# Устройство
# ===========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ===========================
# Загрузка модели
# ===========================
@st.cache_resource
def load_model():
    num_classes = 6
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))

    
    model = model.to(device)
    model.eval()
    return model

model = load_model()
preprocess = MobileNet_V2_Weights.DEFAULT.transforms()
CLASS_NAMES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# ===========================
# Функция предсказания
# ===========================
def predict_image(image: Image.Image):
    start_time = time.time()
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = CLASS_NAMES[output.argmax().item()]
    end_time = time.time()
    inference_time = end_time - start_time
    return pred_class, probs, inference_time

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Intel Image Classification", layout="wide")
st.title("🖼️ Intel Image Classification")

# Вкладки
tab1, tab2, tab3 = st.tabs(["📁 Загрузить файлы", "🔗 По ссылке", "📤 Несколько изображений"])

# --- Вкладка 1: Загрузка файлов ---
with tab1:
    uploaded_files = st.file_uploader("Выберите изображения", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=uploaded_file.name, width=300)
            pred, probs, t = predict_image(image)
            st.success(f"**Предсказание:** {pred} | **Время:** {t:.3f} сек")
            st.bar_chart(dict(zip(CLASS_NAMES, probs)))

# --- Вкладка 2: По ссылке ---
with tab2:
    url = st.text_input("Вставьте URL изображения (должен заканчиваться на .jpg/.png)")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Изображение из URL", width=300)
            pred, probs, t = predict_image(image)
            st.success(f"**Предсказание:** {pred} | **Время:** {t:.3f} сек")
            st.bar_chart(dict(zip(CLASS_NAMES, probs)))
        except Exception as e:
            st.error(f"Не удалось загрузить изображение: {e}")

# --- Вкладка 3: Несколько изображений (массовая обработка) ---
with tab3:
    st.write("Загрузите несколько изображений для пакетной обработки")
    multi_files = st.file_uploader("Массовая загрузка", accept_multiple_files=True, type=["jpg", "jpeg", "png"], key="multi")
    if multi_files:
        total_time = 0.0
        results = []
        for f in multi_files:
            image = Image.open(f).convert("RGB")
            pred, probs, t = predict_image(image)
            total_time += t
            results.append((f.name, pred, t))
        
        st.write(f"✅ Обработано {len(multi_files)} изображений за {total_time:.3f} сек (в среднем: {total_time/len(multi_files):.3f} сек/изображение)")
        
        for name, pred, t in results:
            st.write(f"- **{name}** → `{pred}` ({t:.3f} сек)")

st.divider()
st.caption("Модель: MobileNetV2 (ImageNet → fine-tuned на Intel Image Classification)")