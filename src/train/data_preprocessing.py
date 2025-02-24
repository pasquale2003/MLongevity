import json
import numpy as np
from imblearn.over_sampling import SMOTE

def read_json(file_path):
    """Legge il file JSON e restituisce i dati."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    """Preprocessa i dati e applica SMOTE per bilanciare le classi."""
    processed_data = []
    target = []  # Qui aggiungiamo la variabile target (survival)

    for entry in data:
        # Estrai i dati numerici
        numeric_entry = [
            entry['age'],
            entry['weight'],
            entry['height'],
            entry['sys_bp'],
            entry['num_meds'],
            entry['occup_danger'],
            entry['ls_danger'],
            entry['drinks_aweek'],
            entry['major_surgery_num'],
            entry['cholesterol'],
        ]

        # Trasformazione delle variabili categoriche in numeri
        sex = 1 if entry['sex'] == 'm' else 0
        smoker = 1 if entry['smoker'] == 'y' else 0
        nic_other = 1 if entry['nic_other'] == 'y' else 0
        cannabis = 1 if entry['cannabis'] == 'y' else 0
        opioids = 1 if entry['opioids'] == 'y' else 0
        other_drugs = 1 if entry['other_drugs'] == 'y' else 0
        addiction = 1 if entry['addiction'] == 'y' else 0
        diabetes = 1 if entry['diabetes'] == 'y' else 0
        hds = 1 if entry['hds'] == 'y' else 0
        asthma = 1 if entry['asthma'] == 'y' else 0
        immune_defic = 1 if entry['immune_defic'] == 'y' else 0
        family_cancer = 1 if entry['family_cancer'] == 'y' else 0
        family_heart_disease = 1 if entry['family_heart_disease'] == 'y' else 0
        family_cholesterol = 1 if entry['family_cholesterol'] == 'y' else 0

        # Logica per determinare la variabile 'survival'
        if entry['age'] > 80 and entry['diabetes'] == 'n' and entry['asthma'] == 'n' and entry['family_heart_disease'] == 'n':
            survival = 1  # Alta sopravvivenza
        else:
            survival = 0  # Bassa sopravvivenza

        # Aggiungi i dati preprocessati e il target
        processed_data.append(numeric_entry + [sex, smoker, cannabis, opioids, other_drugs, addiction, asthma, immune_defic, hds, nic_other, diabetes, family_cancer, family_heart_disease, family_cholesterol])
        target.append(survival)

    # Converte in array NumPy
    X = np.array(processed_data)
    y = np.array(target)

    # Applica SMOTE per bilanciare le classi
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    return X_res, y_res

# Carica i dati JSON
data = read_json('../data/data.json')

# Preprocessa i dati e applica SMOTE
X_res, y_res = preprocess_data(data)

# Salva i dati bilanciati in un file .npz
np.savez('../data/processed_data.npz', X_res=X_res, y_res=y_res)

# Stampa i risultati per verificare l'equilibrio delle classi
print(f"Dimensioni dopo SMOTE - X_res: {X_res.shape}, y_res: {y_res.shape}")
print(f"Distribuzione delle classi dopo SMOTE: {np.bincount(y_res)}")
