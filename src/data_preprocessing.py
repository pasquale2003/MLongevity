import json
import numpy as np

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def preprocess_data(data):
    processed_data = []

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

        # Trasformare le variabili categoriche in numeri
        sex = 1 if entry['sex'] == 'm' else 0  # m -> 1, f -> 0
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

        # Aggiungi tutte le variabili trasformate
        processed_data.append(numeric_entry + [sex, smoker, cannabis, opioids, other_drugs, addiction, asthma, immune_defic, hds, nic_other, diabetes, family_cancer, family_heart_disease, family_cholesterol])

    return np.array(processed_data)

# Carica i dati JSON
data = read_json('dataset.json')

# Preprocessa i dati
processed_data = preprocess_data(data)

# Stampa il risultato
print(processed_data)
