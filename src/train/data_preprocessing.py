import json
import numpy as np


def read_json(file_path):
    """Legge il file JSON e restituisce i dati."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def preprocess_data(data):
    """Preprocessa i dati per la regressione dell'età alla morte."""
    processed_data = []
    target = []  # L'età alla morte sarà la nostra variabile target

    for entry in data:
        # Estrai i dati numerici
        numeric_entry = [
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

        # Aggiungi i dati preprocessati
        processed_data.append(numeric_entry + [
            sex, smoker, cannabis, opioids, other_drugs, addiction, asthma, immune_defic,
            hds, nic_other, diabetes, family_cancer, family_heart_disease, family_cholesterol
        ])

        # La variabile target è direttamente l'età alla morte
        target.append(entry['age'])

    # Converte in array NumPy
    X = np.array(processed_data)
    y = np.array(target)

    return X, y


# Carica i dati JSON
data = read_json('../data/data.json')

# Preprocessa i dati
X, y = preprocess_data(data)

# Salva i dati preprocessati in un file .npz
np.savez('../data/processed_data.npz', X=X, y=y)

# Stampa i risultati per verificare le dimensioni
print(f"Dimensioni del dataset - X: {X.shape}, y: {y.shape}")
