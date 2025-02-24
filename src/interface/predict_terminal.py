import joblib
import numpy as np


# Funzione di preprocessamento dei dati (senza conversioni)
def preprocess_single_data(entry):
    # Estrai i dati numerici
    numeric_entry = [
        entry['age'],
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
    # Raccolta dei dati da terminale
    age = int(input("Inserisci età: "))
    weight_lb = float(input("Inserisci peso (in libbre): "))  # Peso in libbre
    height_inch = float(input("Inserisci altezza (in pollici): "))  # Altezza in pollici
    sys_bp = int(input("Inserisci pressione sistolica (mmHg): "))
    num_meds = int(input("Inserisci il numero di farmaci che prendi: "))

    # Modificato per valori tra 1 e 3
    occup_danger = int(input("La tua occupazione è pericolosa (1 per basso, 2 per medio, 3 per alto): "))
    while occup_danger not in [1, 2, 3]:
        print("Valore non valido. Devi inserire 1, 2 o 3.")
        occup_danger = int(input("La tua occupazione è pericolosa (1 per basso, 2 per medio, 3 per alto): "))

    ls_danger = int(input("Il tuo stile di vita è pericoloso (1 per basso, 2 per medio, 3 per alto): "))
    while ls_danger not in [1, 2, 3]:
        print("Valore non valido. Devi inserire 1, 2 o 3.")
        ls_danger = int(input("Il tuo stile di vita è pericoloso (1 per basso, 2 per medio, 3 per alto): "))

    drinks_aweek = int(input("Quante bevute alla settimana? "))
    major_surgery_num = int(input("Hai subito interventi chirurgici importanti (numero): "))
    cholesterol = int(input("Inserisci livello di colesterolo: "))
    sex = input("Inserisci sesso (m per maschio, f per femmina): ").lower()
    smoker = input("Sei fumatore? (y per sì, n per no): ").lower()
    nic_other = input("Usi altre sostanze nicotiniche? (y per sì, n per no): ").lower()
    cannabis = input("Usi cannabis? (y per sì, n per no): ").lower()
    opioids = input("Usi oppioidi? (y per sì, n per no): ").lower()
    other_drugs = input("Usi altre droghe? (y per sì, n per no): ").lower()
    addiction = input("Hai dipendenze? (y per sì, n per no): ").lower()
    diabetes = input("Hai il diabete? (y per sì, n per no): ").lower()
    hds = input("Hai una storia familiare di malattie gravi? (y per sì, n per no): ").lower()
    asthma = input("Hai l'asma? (y per sì, n per no): ").lower()
    immune_defic = input("Hai una deficienza immunitaria? (y per sì, n per no): ").lower()
    family_cancer = input("Hai una storia familiare di cancro? (y per sì, n per no): ").lower()
    family_heart_disease = input("Hai una storia familiare di malattie cardiache? (y per sì, n per no): ").lower()
    family_cholesterol = input("Hai una storia familiare di colesterolo alto? (y per sì, n per no): ").lower()

    # Organizza i dati in un dizionario
    user_data = {
        'age': age,
        'weight': weight_lb,  # Peso in libbre
        'height': height_inch,  # Altezza in pollici
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


# Carica il modello salvato
model = joblib.load('../model/mlongevity_model.pkl')

# Raccogli i dati dell'utente
user_data = collect_user_data()

# Preprocessa i dati dell'utente
processed_user_data = preprocess_single_data(user_data)

# Fai la previsione
prediction = model.predict(processed_user_data)

# Stampa la previsione
if prediction[0] == 1:
    print("Previsione: longevità più alta della media.")
else:
    print("Previsione: longevità più bassa della media.")
