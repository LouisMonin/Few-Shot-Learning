import json
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


import sys
import numpy as np

# Modifier l'encodage par défaut pour UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Charger le tokenizer et le modèle BERT pré-entraînés
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Charger le fichier JSON
with open("file_corrige_with_ids.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Structures pour organiser les vecteurs par label
desc_vectors = {0: [], 1: [], 2: [], 3: [], 4: []}
title_vectors = {0: [], 1: [], 2: [], 3: [], 4: []}

def get_text_embedding(text):
    # Tokeniser le texte et convertir en IDs de tokens
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Obtenir les sorties du modèle
    with torch.no_grad():
        outputs = model(**inputs)
    # Utiliser l'embedding du token [CLS] ou faire la moyenne des embeddings des tokens
    embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze()

# Collecte des vecteurs
for item in data:
    label = item.get("label")
    desc = item.get("description")
    title = item.get("jobTitle")

    # Vérifiez et traitez la description
    if desc and isinstance(desc, str) and desc.strip().lower() not in ["none", "nan"] and label in [0.0, 1.0, 2.0, 3.0, 4.0]:
        desc_vect = get_text_embedding(desc)
        desc_vectors[label].append(desc_vect)

    # Vérifiez et traitez le titre
    if title and isinstance(title, str) and title.strip().lower() not in ["none", "nan"] and label in [0.0, 1.0, 2.0, 3.0, 4.0]:
        title_vect = get_text_embedding(title)
        title_vectors[label].append(title_vect)

# Calcul des vecteurs moyens et préparation pour la sérialisation JSON
prototypes = {"desc": {}, "title": {}}
for label in desc_vectors:
    if desc_vectors[label]:  # S'assurer que la liste n'est pas vide
        desc_tensors = torch.stack(desc_vectors[label])
        desc_mean = torch.mean(desc_tensors, dim=0)
        prototypes["desc"][str(label)] = desc_mean.tolist()
    if title_vectors[label]:  # S'assurer que la liste n'est pas vide
        title_tensors = torch.stack(title_vectors[label])
        title_mean = torch.mean(title_tensors, dim=0)
        prototypes["title"][str(label)] = title_mean.tolist()

# Sauvegarde des prototypes dans un fichier JSON
with open("prototypes_Bert.json", "w") as json_file:
    json.dump(prototypes, json_file)

print("Vecteurs moyens sauvegardés dans 'prototypes.json'.")

