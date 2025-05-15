import json
import torch
import nltk
import spacy
from transformers import GPT2Model, GPT2Tokenizer
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense
from kerastuner.tuners import RandomSearch
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from langdetect import detect
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import sys
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer




# Modifier l'encodage par défaut pour UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Charger le modèle et le tokenizer GPT-3
model_name = "gpt2"  # ou tout autre modèle GPT disponible (gpt2-medium, gpt2-large, etc.)
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Charger le fichier JSON
with open("file_corrigé_with_ids_gpt.json", "r", encoding="utf-8") as file:
    data = json.load(file)


def vectoriser_text(text):
    text = text.replace("\u2028", "").replace("\u2029", "")
    print("Texte brut: ", text)

    try:
        langue = detect(text)
    except:
        langue = "fr"  # Langue par défaut en cas d'échec de détection

    if langue == "fr":
        nlp = spacy.load("fr_core_news_sm")
    elif langue == "en":
        nlp = spacy.load("en_core_web_sm")
    else:
        nlp = spacy.load("fr_core_news_sm")  # Fallback sur le modèle multilingue

    doc = nlp(text)
    tokens_nettoyes = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and not token.like_num and not token.is_space]

    texte_nettoye = " ".join(tokens_nettoyes)

    print("Texte nettoye: ", texte_nettoye)

    # Tokeniser le texte
    if len(texte_nettoye) > 0:
        tokens = tokenizer.encode(texte_nettoye, return_tensors='pt')

    # Passer les tokens au modèle pour obtenir les représentations
    if tokens.numel() > 0:
        with torch.no_grad():
            outputs = model(tokens)
    else:
        print("Erreur: Les tokens sont vides.")
        return 0

    # Obtenir les représentations cachées de chaque token
    if 'outputs' in locals():
        hidden_states = outputs.last_hidden_state
        text_embedding = hidden_states.mean(dim=1).squeeze()
        return text_embedding  # hidden_states contient les représentations cachées
    else:
        # Gérez le cas où 'outputs' n'est pas défini.
        print("Erreur: 'outputs' n'est pas défini.")
        return None  # Ou une valeur par défaut.

    # Afficher la forme de hidden_states
    #print("Forme des representations cachees:", hidden_states.shape)

    # Obtenir la représentation vectorielle pour l'ensemble du texte
    text_embedding = outputs[0].mean(dim=1).squeeze()
    return text_embedding


desc_list = []
label_list = []
desc_vect_list = []

id = 6297

desc_list_test = []
label_list_test = []
desc_vect_list_test = []
# with open("desc+title_vect_list_clean.json", 'a', encoding="utf-8") as fichier:
#     for item in data[9647:]:
#         label = item.get("label")
#         desc = item.get("description")
#         title = item.get("jobTitle")
#         print("Description: ", desc)
#         print("Titre: ", title)
#         print("Label: ", label)
#         if (label is not None and label in [0, 1, 2, 3, 4]):
#             label_list.append(label)
#             # Remove unusual line terminator characters
#             if desc is not None and isinstance(desc, str) and desc.strip().lower() != "none" and desc.strip().lower() != "nan" :
#                 if title is not None and isinstance(title, str) and title.strip().lower() != "none" and title.strip().lower() != "nan" :
#                     desc_vect = vectoriser_text(title + " " + desc)
#                 else:
#                     desc_vect = vectoriser_text(desc)
#             else:
#                 if title is not None and isinstance(title, str) and title.strip().lower() != "none" and title.strip().lower() != "nan" :
#                     desc_vect = vectoriser_text(title)
#                 else:
#                     desc_vect = None
#             if desc_vect is not None:
#                 desc_vect_list.append(desc_vect)
#                 # Convertir le Tensor en un tableau NumPy
#                 desc_vect_numpy = desc_vect.numpy()
#                 # Écrire les données dans le fichier JSON
#                 json.dump(
#                     {
#                         "title": title,
#                         "description": desc,
#                         "label": label,
#                         "id": id,
#                         "vecteur": desc_vect_numpy.tolist(),
#                     },
#                     fichier,
#                     ensure_ascii=False,
#                 )
#                 fichier.write("\n")
#                 id += 1
#             else: # Gérer le cas où la vectorisation échoue
#                 print("Erreur: Aucune titre ou description n'a été détectée.")




# Charger le fichier JSON et préparer les données
desc_vect_list = []
label_list = []
desc_list = []
with open("desc_vect_list_clean.json", "r", encoding="utf-8") as file:
    vecteurs_data = json.load(file)
    for item in vecteurs_data:
        label = item.get("label")
        vecteur = item.get("vecteur")
        desc_vect_list.append(vecteur)
        label_list.append(label)
        desc_list.append(item.get("description"))

X = np.array(desc_vect_list)
y = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
        units=hp.Int('units', min_value=32, max_value=512, step=32),
        activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=25,
    executions_per_trial=1,
    directory='my_dir',
    project_name='my_project_only_desc')

tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Obtenir le meilleur modèle
best_model = tuner.get_best_models(num_models=1)[0]

# Prédire les classes sur l'ensemble de test avec le meilleur modèle
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculer et afficher le score F1, la matrice de confusion, et le rapport de classification
print("Score F1:", f1_score(y_test, y_pred_classes, average='weighted'))
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

# Évaluation du meilleur modèle
loss, accuracy = best_model.evaluate(X_test, y_test)
print("Accuracy du meilleur modèle:", accuracy)