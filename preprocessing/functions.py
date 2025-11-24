import re
import pandas as pd
import spacy
from spacy.lang.es.stop_words import STOP_WORDS as stopwords


def expresiones_regulares(texto: str) -> str:
    # Remove accents/tildes but keep the letters
    texto = re.sub(r'[áàäâã]', 'a', texto)
    texto = re.sub(r'[éèëê]', 'e', texto)
    texto = re.sub(r'[íìïî]', 'i', texto)
    texto = re.sub(r'[óòöôõ]', 'o', texto)
    texto = re.sub(r'[úùüû]', 'u', texto)
    texto = re.sub(r'[ÁÀÄÂÃ]', 'A', texto)
    texto = re.sub(r'[ÉÈËÊ]', 'E', texto)
    texto = re.sub(r'[ÍÌÏÎ]', 'I', texto)
    texto = re.sub(r'[ÓÒÖÔÕ]', 'O', texto)
    texto = re.sub(r'[ÚÙÜÛ]', 'U', texto)

    # Convert to lowercase, remove non-alphabetic characters except spaces and ñ/ü, normalize spaces
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zñü ]', '', texto.lower())).strip()


def tokenizar(texto: str) -> str:
    nlp = spacy.load("es_core_news_lg")

    stopwords_personalizados = [
        "medico", "paciente", "psicologo", "psicologa",
        "psicologia", "psicoterapeuta", "psicoterapia", "refiere"
    ]
    all_stopwords = stopwords.union(stopwords_personalizados)

    tokens = [
        token.text for token in nlp(texto)
        if token.text.lower() not in all_stopwords and not token.is_punct and not token.is_space
    ]

    return " ".join(tokens)


def lematizar(texto: str) -> str:
    nlp = spacy.load("es_core_news_lg")

    tokens = texto.split()
    lemmas = [token.lemma_ for token in nlp(" ".join(tokens))]

    return " ".join(lemmas)
