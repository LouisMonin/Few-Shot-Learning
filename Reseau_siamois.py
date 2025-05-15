import pandas as pd  # Importation de la bibliothèque pandas pour la manipulation des données
import numpy as np  # Importation de la bibliothèque numpy pour les opérations numériques
import matplotlib.pyplot as plt  # Importation de la bibliothèque matplotlib pour la visualisation des données
from sklearn.model_selection import train_test_split  # Importation de train_test_split pour diviser les données
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder  # Importation de différents prétraitement des données
from sklearn.metrics import classification_report  # Importation de classification_report pour évaluer la performance du modèle
from tensorflow.keras.models import Model  # Importation de Model pour la construction du modèle
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten  # Importation des couches du modèle
from tensorflow.keras.optimizers import Adam  # Importation de l'optimiseur Adam
from tensorflow.keras.utils import to_categorical  # Importation pour convertir les labels en format catégorique

# Charger et prétraiter les données
data = pd.read_csv('linkedin_profiles.csv')  # Charger les données à partir d'un fichier CSV
data = data[['companyName', 'jobTitle', 'location', 'description', 'label']].dropna()  # Sélectionner les colonnes pertinentes et supprimer les valeurs manquantes

label_encoder = LabelEncoder().fit(data['label'].astype(str))  # Initialisation de l'encodeur de labels et ajustement sur les données
data['label'] = label_encoder.fit_transform(data['label'])  # Transformer les labels en entiers

categorical_cols = ['companyName', 'jobTitle', 'location']  # Liste des colonnes catégorielles dans les données

# Fonction de prétraitement des données
def preprocess_data(df, categorical_cols, scaler=None, ohe=None, fit_transform=True):
    """
    Prétraite les données en effectuant la normalisation, l'encodage one-hot et d'autres transformations.

    Args:
        df (DataFrame): Le DataFrame contenant les données à prétraiter.
        categorical_cols (list): Liste des noms de colonnes catégorielles dans les données.
        scaler: L'objet de normalisation à utiliser.
        ohe: L'objet d'encodage one-hot à utiliser.
        fit_transform (bool): Indique s'il faut adapter et transformer les données.

    Returns:
        DataFrame: Le DataFrame prétraité.
        scaler: L'objet de normalisation utilisé.
        ohe: L'objet d'encodage one-hot utilisé.
    """
    if 'description' in df.columns:  # Vérifier si la colonne 'description' est présente
        df['description_len'] = df['description'].apply(len)  # Calculer la longueur de la description
        df = df.drop(['description'], axis=1)  # Supprimer la colonne 'description'

    df_non_cat = df.drop(categorical_cols, axis=1)  # Séparer les colonnes non catégorielles
    df_cat = df[categorical_cols]  # Sélectionner uniquement les colonnes catégorielles

    if fit_transform:  # Si l'adaptation et la transformation doivent être effectuées
        if scaler is None:  # Vérifier si l'objet de normalisation est fourni
            scaler = StandardScaler()  # Initialiser un objet de normalisation
        if ohe is None:  # Vérifier si l'objet d'encodage one-hot est fourni
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')  # Initialiser un objet d'encodage one-hot
        
        df_cat_encoded = ohe.fit_transform(df_cat)  # Encodage one-hot des colonnes catégorielles
        df_non_cat['description_len'] = scaler.fit_transform(df_non_cat[['description_len']])  # Normalisation de la longueur de la description
    else:
        df_cat_encoded = ohe.transform(df_cat)  # Encodage one-hot sans ajustement
        df_non_cat['description_len'] = scaler.transform(df_non_cat[['description_len']])  # Normalisation sans ajustement
    
    df_cat_encoded = pd.DataFrame(df_cat_encoded, index=df.index, columns=ohe.get_feature_names_out())  # Création d'un DataFrame à partir de l'encodage one-hot

    final_df = pd.concat([df_non_cat.reset_index(drop=True), df_cat_encoded.reset_index(drop=True)], axis=1)  # Concaténation des colonnes non catégorielles et encodées

    return final_df, scaler, ohe  # Retourner les données prétraitées et les objets de normalisation/encodage one-hot

data, scaler, ohe = preprocess_data(data, categorical_cols, fit_transform=True)  # Appel de la fonction preprocess_data pour prétraiter les données

X = data.drop('label', axis=1).values  # Sélectionner les features
y = data['label'].values  # Sélectionner les labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Diviser les données en ensembles d'entraînement et de test

y_train_cat = to_categorical(y_train, num_classes=5)  # Convertir les labels d'entraînement en format catégorique
y_test_cat = to_categorical(y_test, num_classes=5)  # Convertir les labels de test en format catégorique

# Créer le réseau de base
def create_base_network_multiclass(input_shape):
    """
    Crée un réseau de neurones artificiels de base pour la classification multiclasse.

    Args:
        input_shape (tuple): La forme de l'entrée du réseau.

    Returns:
        Model: Le modèle de réseau de neurones.
    """
    input = Input(shape=input_shape)  # Création d'une couche d'entrée avec la forme spécifiée
    x = Dense(128, activation='relu')(input)  # Couche dense avec 128 neurones et fonction d'activation ReLU
    x = Dropout(0.1)(x)  # Dropout avec un taux de 0.1 pour la régularisation
    x = Dense(128, activation='relu')(x)  # Couche dense avec 128 neurones et fonction d'activation ReLU
    x = Dropout(0.1)(x)  # Dropout avec un taux de 0.1 pour la régularisation
    x = Flatten()(x)  # Aplatir les données pour les passer à une couche Dense
    x = Dense(5, activation='softmax')(x)  # Couche dense de sortie avec 5 neurones pour la classification multiclasse et fonction d'activation softmax
    return Model(input, x)  # Création du modèle avec l'entrée et la sortie spécifiées

input_shape = X_train.shape[1:]  # Récupération de la forme de l'entrée
base_network_multiclass = create_base_network_multiclass(input_shape)  # Création du modèle de réseau de base
base_network_multiclass.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # Compilation du modèle avec l'optimiseur Adam, la fonction de perte d'entropie croisée catégorique et la métrique d'exactitude

base_network_multiclass.fit(X_train, y_train_cat, epochs=20, batch_size=128, validation_split=0.2)  # Entraînement du modèle sur les données d'entraînement avec une validation de 20%

loss, accuracy = base_network_multiclass.evaluate(X_test, y_test_cat)  # Évaluation du modèle sur les données de test
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')  # Affichage de la perte et de l'exactitude du test

# Visualisation du nombre d'éléments dans chaque classe de label
label_counts = data['label'].value_counts()  # Compter le nombre d'éléments dans chaque classe de label
plt.bar(label_encoder.inverse_transform(label_counts.index), label_counts.values)  # Créer un diagramme à barres pour visualiser les données
plt.xlabel('Label')  # Ajouter une étiquette à l'axe des abscisses
plt.ylabel('Nombre d\'éléments')  # Ajouter une étiquette à l'axe des ordonnées
plt.title('Nombre d\'éléments dans chaque classe de label')  # Ajouter un titre au diagramme
plt.show()  # Afficher le diagramme

# Calcul du F1-score
y_pred = np.argmax(base_network_multiclass.predict(X_test), axis=1)  # Prédire les labels pour les données de test
print("Rapport de classification:")  # Afficher un titre pour le rapport de classification
# Convertir les noms de classe en chaînes de caractères
class_names = [str(cls) for cls in label_encoder.classes_]
# Afficher le rapport de classification avec les noms de classe convertis
print(classification_report(y_test, y_pred, target_names=class_names))

# Fonction pour classer un nouveau profil
def classify_new_profile(new_profile, model):
    """
    Classe un nouveau profil en utilisant un modèle de réseau de neurones.

    Args:
        new_profile (array-like): Le nouveau profil à classer.
        model (Model): Le modèle de réseau de neurones à utiliser pour la classification.

    Returns:
        int: La classe prédite pour le nouveau profil.
    """
    new_profile_expanded = np.expand_dims(new_profile, axis=0)  # Expansion des dimensions pour une seule observation
    predictions = model.predict(new_profile_expanded)  # Prédiction de la classe du nouveau profil
    predicted_class = np.argmax(predictions)  # Sélection de la classe prédite avec la plus haute probabilité
    return predicted_class  # Retourne la classe prédite pour le nouveau profil

# Exemple de classification d'un nouveau profil
nouveau_profil = {
    'companyName': ['Google'],
    'jobTitle': ['Data Analyst'],
    'location': ['Mountain'],
    'description_len': [120]
}

nouveau_profil_df = pd.DataFrame(nouveau_profil)  # Création d'un DataFrame à partir du nouveau profil

# Prétraiter le nouveau profil
nouveau_profil_processed, _, _ = preprocess_data(nouveau_profil_df, categorical_cols, scaler, ohe, fit_transform=False)  # Prétraitement du nouveau profil

nouveau_profil_processed_flat = nouveau_profil_processed.to_numpy().flatten()  # Aplatir le nouveau profil pour l'entrée du modèle
classification = classify_new_profile(nouveau_profil_processed_flat, base_network_multiclass)  # Classification du nouveau profil
print("Classification du nouveau profil:", label_encoder.inverse_transform([classification])[0])  # Affichage de la classe prédite pour le nouveau profil
