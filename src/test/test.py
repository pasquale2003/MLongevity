import json
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Funzione per caricare i dati JSON di test
def load_test_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Funzione per preprocessare i dati nello stesso formato del training
def preprocess_test_data(data):
    processed_data = []
    true_ages = []  # Lista per conservare l'età reale

    for entry in data:
        # Estrai i dati numerici
        numeric_entry = [
            entry['weight'], entry['height'], entry['sys_bp'],
            entry['num_meds'], entry['occup_danger'], entry['ls_danger'],
            entry['drinks_aweek'], entry['major_surgery_num'], entry['cholesterol']
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

        # Costruisci la feature list
        processed_data.append(numeric_entry + [sex, smoker, cannabis, opioids,
                                               other_drugs, addiction, asthma, immune_defic,
                                               hds, nic_other, diabetes, family_cancer,
                                               family_heart_disease, family_cholesterol])

        # Salva l'età reale
        true_ages.append(entry['age'])

    return np.array(processed_data), np.array(true_ages)

# Carica il modello
model = joblib.load('../model/mlongevity_model.pkl')

# Carica i dati di test
test_data = load_test_data('../data/data_test.json')  # Sostituisci con il tuo file JSON di test

# Preprocessa i dati di test
X_test, true_ages = preprocess_test_data(test_data)

# Effettua la previsione
predicted_ages = model.predict(X_test)

# Stampa le previsioni rispetto ai valori reali
print("\n📌 Confronto tra età predetta e reale:\n")
for i in range(len(true_ages)):
    print(f"👤 Persona {i+1}: Predetto = {predicted_ages[i]:.2f}, Reale = {true_ages[i]}")

# Calcola metriche di errore
mae = mean_absolute_error(true_ages, predicted_ages)
mse = mean_squared_error(true_ages, predicted_ages)
r2 = r2_score(true_ages, predicted_ages)

print("\n📊 **Metriche del modello sui dati di test**:")
print(f"🔹 Mean Absolute Error (MAE): {mae:.2f}")
print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
print(f"🔹 R² Score: {r2:.2f}")
