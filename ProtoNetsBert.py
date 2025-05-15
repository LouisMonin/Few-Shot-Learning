import json
import torch
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F


import sys

# Modifier l'encodage par défaut pour UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Charger le tokenizer et le modèle BERT pré-entraînés
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Charger le fichier JSON
with open("file_corrige_with_ids.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Liste pour stocker les descriptions et les identifiants
items_0 = []
items_1 = []
items_2 = []
items_3 = []
items_4 = []
items_sans_label = []

descriptions_0 = []
descriptions_1 = []
descriptions_2 = []
descriptions_3 = []
descriptions_4 = []

JobTitles_0 = []
JobTitles_1 = []
JobTitles_2 = []
JobTitles_3 = []
JobTitles_4 = []

data_dict = []

compteurs = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}


# Parcourir les éléments du tableau JSON et extraire les descriptions et les identifiants
for item in data:
    label = item.get("label")
    if label in compteurs and compteurs[label] < 500:
        # Ajoutez un nouvel élément dans `nouveaux_items` au lieu de `data`
        data_dict.append({"label": label, "vecteur_job": None, "vecteur_desc": None, "resultat_distance": -1, "resultat_similarity": -1, "description": item.get("description"), "jobTitle": item.get("jobTitle")})
        compteurs[label] += 1
    else:
        items_sans_label.append(item)

for label, compteur in compteurs.items():
    print(f"éléments avec le label {label}: {compteur}")

print(f"éléments sans label: {len(items_sans_label)}")
        # description = item.get("description")
    # if description is not None and isinstance(description, str) and description.strip().lower() != "none" and description.strip().lower() != "nan":
    #     label = item.get("label")
    #     if label == 0:
    #         descriptions_0.append(description)
    #     elif label == 1:
    #         descriptions_1.append(description)
    #     elif label == 2:
    #         descriptions_2.append(description)
    #     elif label == 3:
    #         descriptions_3.append(description)
    #     elif label == 4:
    #         descriptions_4.append(description)

    # jobTitle = item.get("jobTitle")
    # if jobTitle is not None and isinstance(jobTitle, str) and jobTitle.strip().lower() != "none" and jobTitle.strip().lower() != "nan":
    #     label = item.get("label")
    #     if label == 0:
    #         JobTitles_0.append(jobTitle)
    #         print("Titre label 0", jobTitle)
    #     elif label == 1:
    #         JobTitles_1.append(jobTitle)
    #         print("Titre label 1", jobTitle)
    #     elif label == 2:
    #         JobTitles_2.append(jobTitle)
    #         print("Titre label 2", jobTitle)
    #     elif label == 3:
    #         JobTitles_3.append(jobTitle)
    #         print("Titre label 3", jobTitle)
    #     elif label == 4:
    #         JobTitles_4.append(jobTitle)
    #         print("Titre label 4", jobTitle)

with open("prototypes_Bert.json", "r", encoding="utf-8") as file:
    prototypes = json.load(file)

def parcourir_items(list_items, a):
    print(f"Nombre d'items à traiter : {len(list_items)}")
    résultats_distance = {i: 0 for i in range(5)}  # Initialise les compteurs pour chaque label de 0 à 4
    résultats_similarity = {i: 0 for i in range(5)}  # Initialise les compteurs pour chaque label de 0 à 4
    résultats_corrects_distance = 0
    résultats_corrects_similarity = 0
    résultats_corrects_label_distance = {i: 0 for i in range(5)}
    résultats_corrects_label_similarity = {i: 0 for i in range(5)}


    vecteurs_labels_description = [prototypes["desc"]["0"], prototypes["desc"]["1"], prototypes["desc"]["2"], prototypes["desc"]["3"], prototypes["desc"]["4"]]
    vecteurs_labels_jobTitle = [prototypes["title"]["0"], prototypes["title"]["1"], prototypes["title"]["2"], prototypes["title"]["3"], prototypes["title"]["4"]]

    for item in list_items:
        item["vecteur_desc"] = vectoriser_text(item.get("description"))
        item["vecteur_job"] = vectoriser_text(item.get("jobTitle"))

        distances_description = [calculate_distance(item["vecteur_desc"], vect).item() for vect in vecteurs_labels_description]
        distances_jobTitle = [calculate_distance(item["vecteur_job"], vect).item() for vect in vecteurs_labels_jobTitle]

        similarity_description = [calculate_similarity(item["vecteur_desc"], vect).item() for vect in vecteurs_labels_description]
        similarity_jobTitle = [calculate_similarity(item["vecteur_job"], vect).item() for vect in vecteurs_labels_jobTitle]

        # Combine les distances et trouve le label avec la distance minimale
        distances_combinées = [min(desc, job) for desc, job in zip(distances_description, distances_jobTitle)]
        similarity_combinées = [max(desc, job) for desc, job in zip(similarity_description, similarity_jobTitle)]
        label_pred_distance = distances_combinées.index(min(distances_combinées))
        label_pred_similarity = similarity_combinées.index(max(similarity_combinées))
        item["resultat_distance"] = float(label_pred_distance)
        item["resultat_similarity"] = float(label_pred_similarity)
        résultats_distance[label_pred_distance] += 1
        résultats_similarity[label_pred_similarity] += 1
        if item["resultat_distance"] == item["label"]:
            résultats_corrects_distance += 1
            résultats_corrects_label_distance[item["label"]] += 1
        if item["resultat_similarity"] == item["label"]:
            résultats_corrects_similarity += 1
            résultats_corrects_label_similarity[item["label"]] += 1


        # Décommentez pour du débogage ou des logs détaillés
        print(f"le label est : {item['label']}, le resultat est : {item['resultat_distance']}")

    print(f"Nombre de résultats corrects avec le calcul de distance: {résultats_corrects_distance}")
    for label in range(5):
        print(f"Label {label} trouvés : {résultats_distance[label]} corrects : {résultats_corrects_label_distance.get(label, 0)} attendus : 50")

    print(f"Nombre de résultats corrects avec le calcul de similarité: {résultats_corrects_similarity}")
    for label in range(5):
        print(f"Label {label} trouvés : {résultats_similarity[label]} corrects : {résultats_corrects_label_similarity.get(label, 0)} attendus : 50")

    print(f"le pourcentage de résultats corrects avec le calcul de distance est : {résultats_corrects_distance / len(list_items) * 100}%")
    print(f"le pourcentage de résultats corrects avec le calcul de similarité est : {résultats_corrects_similarity / len(list_items) * 100}%")

    return résultats_corrects_distance, résultats_corrects_similarity


def vectoriser_text(text):
    if text is not None and isinstance(text, str) and text.strip().lower() != "none" and text.strip().lower() != "nan" :
        # Tokeniser le texte
        tokens = tokenizer.encode(text, return_tensors='pt')

        # Passer les tokens au modèle pour obtenir les représentations
            # Tokeniser le texte et convertir en IDs de tokens
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Obtenir les sorties du modèle
        with torch.no_grad():
            outputs = model(**inputs)
        # Utiliser l'embedding du token [CLS] ou faire la moyenne des embeddings des tokens
        embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze()
    else:
        return None

def vectoriser(descriptions):
    # Vérifier si la liste est vide ou nulle
    if not descriptions:
        print("La liste des descriptions est vide.")
        return
    vecteurs = []
    for text in descriptions:
        # Tokeniser le texte
        tokens = tokenizer.encode(text, return_tensors='pt')

        # Passer les tokens au modèle pour obtenir les représentations
        if tokens.numel() > 0:
            with torch.no_grad():
                outputs = model(tokens)
        else:
            print("Erreur: Les tokens sont vides.")

        # Obtenir les représentations cachées de chaque token
        hidden_states = outputs.last_hidden_state  # hidden_states contient les représentations cachées

        # Afficher la forme de hidden_states
        #print("Forme des representations cachees:", hidden_states.shape)

        # Obtenir la représentation vectorielle pour l'ensemble du texte
        text_embedding = outputs[0].mean(dim=1).squeeze()

        # Afficher la forme de la représentation vectorielle
        #print("Forme de l'incorporation de texte:", text_embedding.shape)

        vecteurs.append(text_embedding)
    concatenated_vecteurs = torch.stack(vecteurs)
    vecteur_moyen = torch.mean(concatenated_vecteurs, dim=0)
    print("Forme du vecteur moyen:", vecteur_moyen.shape)
    print ("Vecteur moyen:", vecteur_moyen)

# Calcul de la similarité cosinus
def calculate_similarity(vector, prototypes):
    if vector == None:
        return torch.tensor(10000)
    if isinstance(prototypes, list):
            prototypes = torch.tensor(prototypes)
    similarities = F.cosine_similarity(vector.unsqueeze(0), prototypes, dim=1)
    return similarities

# Calcul de la distance euclidienne
def calculate_distance(vector, prototypes):
    if vector == None:
        return torch.tensor(10000)
    else:
        if isinstance(prototypes, list):
            prototypes = torch.tensor(prototypes)
        distance = torch.norm(vector - prototypes, dim=0)
        return distance

parcourir_items(data_dict, 0)
