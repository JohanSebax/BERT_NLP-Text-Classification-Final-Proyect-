
# Importar librerias necesarias
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn

import os
import numpy as np
import streamlit as st
from preprocessing.functions import expresiones_regulares, tokenizar, lematizar

# Funcion para cargar el modelo entrenado guardado


def load_model(model_path, num_classes=6, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), pretrained_model_name="bert-base-multilingual-cased"):
    class BERTTextClassifier(nn.Module):
        def __init__(self, n_classes):
            super(BERTTextClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(
                pretrained_model_name)
            self.dropout = nn.Dropout(p=0.3)
            self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_out = outputs.pooler_output
            drop_out = self.dropout(pooled_out)  # Aplicar dropout
            output = self.linear(drop_out)  # Capa lineal para la clasificación
            return output

    model = BERTTextClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Funcion para clasificar el texto


def classify_text(text, pretrained_model_name="bert-base-multilingual-cased", device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), max_len=512):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

    return predicted_class.item(), probabilities[0].cpu().numpy()


# Diccionario de mapeo de etiquetas
label_map = {
    0: "Otros Trastornos",
    1: "T. de adaptación",
    2: "T. de ansiedad",
    3: "T. depresivos",
    4: "T. externalizantes",
    5: "T. personalidad"
}

# Modelos Disponibles
model_paths = {
    "BERT": "model/bert_text_classifier_model.pth"
}

st.title("Clasificador de Grupo De Trastornos - Selección de Modelo")

modelo_seleccionado = st.selectbox(
    "Selecciona el modelo para predecir:", list(model_paths.keys()))

texto_usuario = st.text_area("Ingrese el texto clínico a clasificar:")

if st.button("Predecir"):
    if texto_usuario.strip() == "":
        st.warning("Por favor, ingrese un texto válido para la predicción.")
    else:
        # cargar ruta del modelo seleccionado
        model_path = model_paths[modelo_seleccionado]

        if not os.path.exists(model_path):
            st.error(
                f"El modelo seleccionado '{modelo_seleccionado}' no se encuentra en la ruta especificada.")
        else:
            # cargar el modelo
            model = load_model(model_path)

            # Preprocesar el texto
            texto_limpio = expresiones_regulares(texto_usuario)
            texto_tokenizado = tokenizar(texto_limpio)
            texto_lemmatizado = lematizar(texto_tokenizado)
            # Realizar la predicción
            predicted_class, all_probabilities = classify_text(
                texto_lemmatizado)

            st.success(
                f"Clase predicha: {label_map[predicted_class]}")

            st.info(f"Confianza: {all_probabilities[predicted_class]:.4f}")

            # Mostrar probabilidades para todas las clases
            st.subheader("Probabilidad de los Trastornos a diagnosticar:")
            for class_id, prob in enumerate(all_probabilities):
                st.write(
                    f"{label_map[class_id]:25s}: {prob:.4f} ({prob*100:.2f}%)")
