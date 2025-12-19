# pages/1_📊_Информация_о_модели.py
import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import requests
from io import BytesIO
import time
import os

# ----------------------------
# ЗАГРУЗКА МОДЕЛИ
# ----------------------------
@st.cache_resource
def load_model():
    class_names = ["benign", "malignant"]
    model = models.efficientnet_b3()
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names))
    try:
        model.load_state_dict(torch.load("models/effb3_model_low.pth", map_location="cpu"))
    except FileNotFoundError:
        st.error("Файл модели не найден: models/effb3_model_low.pth")
        return None, class_names
    model.eval()
    return model, class_names

model, class_names = load_model()

# Трансформации (должны совпадать с обучением!)
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.760, 0.537, 0.538], std=[0.095, 0.119, 0.133])
])

def predict_image(image: Image.Image) -> tuple:
    start_time = time.time()
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax().item()
        confidence = probs[0, pred_idx].item()
    end_time = time.time()
    return class_names[pred_idx], confidence, end_time - start_time

# ----------------------------
# ЗАГОЛОВОК
# ----------------------------
st.title("📊 Информация о модели")
st.write(
    "**Раннее выявление меланомы — залог успешного лечения и сохранения жизни: при обнаружении на начальной стадии выживаемость превышает 95%. "
    "Меланома — одна из самых агрессивных форм рака кожи, но при своевременной диагностике её можно полностью остановить. "
    "Даже незначительные изменения в форме, цвете или размере родинки могут быть тревожным сигналом, на который важно отреагировать немедленно. "
    "Наша модель помогает вам вовремя заметить подозрительные признаки и вовлечь специалиста на самом раннем этапе. "
    "Помните: забота о себе начинается с внимания к мелочам — ваша кожа говорит с вами, стоит только прислушаться.**"
)

# ----------------------------
# БЛОК ПРЕДСКАЗАНИЯ С ПОДДЕРЖКОЙ НЕСКОЛЬКИХ ИЗОБРАЖЕНИЙ
# ----------------------------
st.markdown("---")
st.header("🔍 Протестируйте модель: загрузите одно или несколько изображений")

input_type = st.radio("Способ загрузки", ["Файл", "URL"], key="demo_input")

images_to_process = []

if input_type == "Файл":
    uploaded_files = st.file_uploader(
        "Выберите изображения (можно несколько)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        key="demo_file"
    )
    if uploaded_files:
        for f in uploaded_files:
            try:
                images_to_process.append((f.name, Image.open(f).convert("RGB")))
            except Exception as e:
                st.error(f"Не удалось открыть {f.name}: {e}")

else:  # URL
    urls_text = st.text_area(
        "Введите URL изображений (по одному на строку)", 
        height=100,
        key="demo_url"
    )
    if urls_text:
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        for i, url in enumerate(urls):
            try:
                response = requests.get(url, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                images_to_process.append((f"url_{i+1}.jpg", image))
            except Exception as e:
                st.error(f"Ошибка загрузки {url}: {e}")

# Обработка и отображение результатов
if images_to_process and model is not None:
    st.subheader(f"Результаты анализа ({len(images_to_process)} изображений)")
    
    # Определяем количество колонок (максимум 3)
    n_cols = min(3, len(images_to_process))
    cols = st.columns(n_cols)
    
    for idx, (name, img) in enumerate(images_to_process):
        with cols[idx % n_cols]:
            st.image(img, caption=name, use_column_width=True)
            with st.spinner("Анализ..."):
                pred_class, confidence, elapsed = predict_image(img)
            st.success(f"**{pred_class}**")
            st.caption(f"Уверенность: {confidence:.2%}")
            st.caption(f"Время: {elapsed:.3f} сек")

elif images_to_process:
    st.error("Модель не загружена — предсказание невозможно")

# ----------------------------
# СТАТИСТИКА И МЕТРИКИ (без изменений)
# ----------------------------
st.markdown("---")
st.header("📂 Состав датасета")
st.write("**Модель была обучена на следующих данных**")
st.write(f"**Классы:** {', '.join(class_names)}")

dataset_stats = {"benign": 1440, "malignant": 1197}
st.write(f"**Всего изображений:** {sum(dataset_stats.values())}")
st.write("**Распределение по классам:**")
st.bar_chart(dataset_stats)

st.header("🔍 Метрики качества")
st.write("**ROC-AUC:** 0.92")
st.write("**Precision:** 0.71")
st.write("**Recall:** 0.97")


# ----------------------------
# CONFUSION MATRIX
# ----------------------------
st.header("🧩 Confusion Matrix")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ваша матрица
cm = np.array([[273, 87],
               [  6, 294]])

# Визуализация
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['benign', 'malignant'],
    yticklabels=['benign', 'malignant'],
    ax=ax
)
ax.set_ylabel('Истинный класс')
ax.set_xlabel('Предсказанный класс')
ax.set_title('Confusion Matrix')

# Отобразить в Streamlit
st.pyplot(fig)