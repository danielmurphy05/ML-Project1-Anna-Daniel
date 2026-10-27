import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler

def columns_drop(data): 
    
    columns_to_drop = ['Id','_PSU','DISPCODE', 'SEQNO', 'IYEAR','IMONTH', 'IDAY', 'IDATE','FMONTH', 'CELLFON3','LANDLINE','SEX', 'PVTRESD2','CSTATE','CTELENUM1','CTELENUM','CTELNUM1','CELLFON2','PVTRESD1', '_VEGETEX', '_VEGRESP', '_FRTRESP', '_FRUITEX' ]
    existing_columns = [col for col in columns_to_drop if col in data.columns]

    if existing_columns:
        # Supprimer les colonnes existantes
        data= data.drop(columns=existing_columns)
        
        print(f"Colonnes supprimées : {', '.join(existing_columns)}")
        
    return data 


def missing_values(data):
    
    # Pourcentage de valeurs manquantes par colonne
    missing_pct = data.isna().mean().sort_values(ascending=False) * 100
    missing_pct = missing_pct[missing_pct > 0]

    threshold= 80 #à modifier pour tester  
    columns_to_drop = missing_pct[missing_pct > threshold].index
    data_clean = data.drop(columns=columns_to_drop, axis=1)
    data_clean = data_clean.fillna(data.median())

    return data_clean
    
    
def binary_col(data):
    binary_cols = [
        col for col in data.columns
        if data[col].nunique(dropna=True) == 2
    ]
    data[binary_cols] = data[binary_cols].replace({2: 0})

    print("Colonnes binaires converties de 1,2 en 1,0.")
    
    return data 

def single_value_col(data):
    # Colonnes qui ne contiennent qu'une seule valeur non manquante
    constant_cols = [
        col for col in data.columns
        if data[col].nunique(dropna=True) <= 1
    ]
    print("Nombre de colonnes constantes :", len(constant_cols))
    data= data.drop(columns=constant_cols)
    
    return data 

def correlation(data):
    data = data.copy()
    corr_matrix = data.corr(numeric_only=True)
    threshold = 0.99

    # Partie triangulaire supérieure, sans la diagonale
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Colonnes à supprimer
    to_drop = [
        col for col in upper.columns
        if (upper[col].abs() > threshold).any()
    ]

    data = data.drop(columns=to_drop)
    print("Colonnes supprimées pour corrélation :", to_drop)

    return data


def preprocessing(data):
    """
    Fonction de prétraitement des données.
    Prend en entrée un DataFrame brut et retourne un DataFrame prétraité.
    """
    # Étape 1 : Supprimer les colonnes inutiles
    data = columns_drop(data)
    
    # Étape 2 : Gérer les valeurs manquantes
    # Supprimer les lignes avec plus de 80% de NaN et Remplir les NaN restants avec la médiane 
    data = missing_values(data)
    
    
    # Étape 3 : Convertir les colonnes binaires (1, 2) en (0, 1)
    data = binary_col(data)
    
    #Remove single value columns
    
    data = single_value_col(data)
    
    # Étape 5 : Supprimer les colonnes fortement corrélées
    data = correlation(data)
    
    # Étape 4 : Standardiser les colonnes numériques
    #numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    #scaler = StandardScaler()
    #data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    
    # Retourner le DataFrame prétraité
    return data

