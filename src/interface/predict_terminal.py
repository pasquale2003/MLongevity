import joblib
import numpy as np

# Funzione di preprocessamento dei dati (senza l'età)
def preprocess_single_data(entry):
    # Estrai i dati numerici
    numeric_entry = [
        entry['weight'],  # Peso in libbre
        entry['height'],  # Altezza in pollici
        entry['sys_bp'],
        entry['num_meds'],
        entry['occup_danger'],
        entry['ls_danger'],
        entry['drinks_aweek'],
        entry['major_surgery_num'],
        entry['cholesterol'],
    ]

    # Trasformare le variabili categoriche in numeri
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

    # Restituisci il vettore con tutte le variabili trasformate
    return np.array([numeric_entry + [sex, smoker, cannabis, opioids, other_drugs, addiction, asthma, immune_defic, hds,
                                      nic_other, diabetes, family_cancer, family_heart_disease, family_cholesterol]])

# Funzione per raccogliere i dati dell'utente
def collect_user_data():
    print("\nInserisci i dati per la previsione:\n")
    try:
        weight_lb = float(input("Inserisci peso (in libbre): "))
        height_inch = float(input("Inserisci altezza (in pollici): "))
        sys_bp = int(input("Inserisci pressione sistolica (mmHg): "))
        num_meds = int(input("Inserisci il numero di farmaci che prendi: "))

        occup_danger = int(input("Pericolosità della tua occupazione (1 = basso, 2 = medio, 3 = alto): "))
        ls_danger = int(input("Pericolosità del tuo stile di vita (1 = basso, 2 = medio, 3 = alto): "))

        drinks_aweek = int(input("Quante bevute fai a settimana? "))
        major_surgery_num = int(input("Quanti interventi chirurgici importanti hai avuto? "))
        cholesterol = int(input("Inserisci livello di colesterolo: "))

        sex = input("Sesso (m per maschio, f per femmina): ").lower()
        smoker = input("Sei fumatore? (y/n): ").lower()
        nic_other = input("Usi altre sostanze nicotiniche? (y/n): ").lower()
        cannabis = input("Usi cannabis? (y/n): ").lower()
        opioids = input("Usi oppioidi? (y/n): ").lower()
        other_drugs = input("Usi altre droghe? (y/n): ").lower()
        addiction = input("Hai dipendenze? (y/n): ").lower()
        diabetes = input("Hai il diabete? (y/n): ").lower()
        hds = input("Storia familiare di malattie gravi? (y/n): ").lower()
        asthma = input("Hai l'asma? (y/n): ").lower()
        immune_defic = input("Hai una deficienza immunitaria? (y/n): ").lower()
        family_cancer = input("Storia familiare di cancro? (y/n): ").lower()
        family_heart_disease = input("Storia familiare di malattie cardiache? (y/n): ").lower()
        family_cholesterol = input("Storia familiare di colesterolo alto? (y/n): ").lower()

        user_data = {
            'weight': weight_lb,
            'height': height_inch,
            'sys_bp': sys_bp,
            'num_meds': num_meds,
            'occup_danger': occup_danger,
            'ls_danger': ls_danger,
            'drinks_aweek': drinks_aweek,
            'major_surgery_num': major_surgery_num,
            'cholesterol': cholesterol,
            'sex': sex,
            'smoker': smoker,
            'nic_other': nic_other,
            'cannabis': cannabis,
            'opioids': opioids,
            'other_drugs': other_drugs,
            'addiction': addiction,
            'diabetes': diabetes,
            'hds': hds,
            'asthma': asthma,
            'immune_defic': immune_defic,
            'family_cancer': family_cancer,
            'family_heart_disease': family_heart_disease,
            'family_cholesterol': family_cholesterol
        }

        return user_data
    except ValueError:
        print("\nErrore: Inserisci solo numeri validi per i valori numerici!\n")
        return collect_user_data()

# Carica il modello salvato
model = joblib.load('../model/mlongevity_model.pkl')

# Raccogli i dati dell'utente
user_data = collect_user_data()

# Preprocessa i dati dell'utente
processed_user_data = preprocess_single_data(user_data)

# Fai la previsione
predicted_age = model.predict(processed_user_data)[0]

# Stampa la previsione dell'età
print("\n**Previsione della longevità**:")
print(f"Età stimata alla morte: {predicted_age:.2f} anni")