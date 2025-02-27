import json
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import random  # Aggiunto per il campionamento casuale

def load_data(file_path):
    """Carica il file JSON e restituisce i dati."""
    with open(file_path, 'r') as file:
        return json.load(file)

def count_instances(data):
    """Conta il numero di istanze nel dataset."""
    return len(data)

def calculate_average(data, key):
    """Calcola la media di un dato numerico specifico."""
    values = [instance[key] for instance in data if isinstance(instance.get(key), (int, float))]
    return sum(values) / len(values) if values else 0

def count_categorical(data, key, value):
    """Conta il numero di istanze con un valore specifico in una chiave."""
    return sum(1 for instance in data if instance.get(key) == value)

def count_distribution(data, key):
    """Conta la distribuzione di valori per una determinata chiave."""
    return Counter(instance.get(key) for instance in data if key in instance)

def analyze_target_distribution(data, target='age'):
    """Analizza la distribuzione della variabile target (età alla morte)."""

    # Estrai i valori della variabile target
    target_values = [instance[target] for instance in data if isinstance(instance.get(target), (int, float))]

    # Calcola skewness (asimmetria) e kurtosis (curtosi)
    target_skewness = skew(target_values)
    target_kurtosis = kurtosis(target_values)

    # Istogramma della distribuzione
    plt.figure(figsize=(10, 5))
    plt.hist(target_values, bins=30, color='blue', alpha=0.7)
    plt.axvline(x=sum(target_values) / len(target_values), color='red', linestyle='dashed', label='Media')
    plt.title(f'Distribuzione della variabile target ({target})')
    plt.xlabel(target)
    plt.ylabel('Frequenza')
    plt.legend()
    plt.show()

    # Boxplot per identificare outlier
    plt.figure(figsize=(8, 4))
    plt.boxplot(target_values, vert=False, patch_artist=True, boxprops=dict(facecolor='blue'))
    plt.title(f'Boxplot della variabile target ({target})')
    plt.show()

    # Stampa i risultati
    print(f"\nAnalisi della distribuzione di {target}:")
    print(f"- Skewness (Asimmetria): {target_skewness:.2f}")
    print(f"- Kurtosis (Curtosi): {target_kurtosis:.2f}")

    if target_skewness > 1:
        print("📌 La distribuzione è positivamente asimmetrica (coda lunga a destra).")
    elif target_skewness < -1:
        print("📌 La distribuzione è negativamente asimmetrica (coda lunga a sinistra).")
    else:
        print("✅ La distribuzione è abbastanza simmetrica.")

    if target_kurtosis > 3:
        print("📌 La distribuzione ha code più pesanti rispetto a una normale (più outlier).")
    elif target_kurtosis < 3:
        print("✅ La distribuzione ha code più leggere rispetto a una normale.")

# Nuova funzione per calcolare la media dell'età di morte per fumatori e non fumatori
def calculate_smoking_groups_average(data, smoker_key='smoker', death_age_key='age', sample_size=200):
    """Calcola la media dell'età di morte per fumatori e non fumatori, usando un campionamento casuale."""

    # Filtra fumatori e non fumatori
    smokers = [instance for instance in data if instance.get(smoker_key) == 'y']
    non_smokers = [instance for instance in data if instance.get(smoker_key) == 'n']

    # Verifica che ci siano abbastanza dati per il campionamento
    if len(smokers) >= sample_size and len(non_smokers) >= sample_size:
        # Campiona casualmente 200 fumatori e 200 non fumatori
        sampled_smokers = random.sample(smokers, sample_size)
        sampled_non_smokers = random.sample(non_smokers, sample_size)

        # Calcola la media dell'età di morte per i fumatori
        avg_death_age_smokers = calculate_average(sampled_smokers, death_age_key)
        # Calcola la media dell'età di morte per i non fumatori
        avg_death_age_non_smokers = calculate_average(sampled_non_smokers, death_age_key)

        return avg_death_age_smokers, avg_death_age_non_smokers
    else:
        print("Non ci sono abbastanza dati per eseguire il campionamento.")
        return None, None

def main():
    # Caricare i dati dal file JSON
    data = load_data('data.json')

    # Eseguire l'analisi
    num_instances = count_instances(data)
    avg_age = calculate_average(data, 'age')
    avg_weight = calculate_average(data, 'weight')
    avg_height = calculate_average(data, 'height')
    avg_sys_bp = calculate_average(data, 'sys_bp')
    avg_cholesterol = calculate_average(data, 'cholesterol')
    avg_drinks = calculate_average(data, 'drinks_aweek')
    avg_surgeries = calculate_average(data, 'major_surgery_num')

    num_smokers = count_categorical(data, 'smoker', 'y')
    num_men = count_categorical(data, 'sex', 'm')
    num_women = count_categorical(data, 'sex', 'f')
    num_people_on_meds = sum(1 for instance in data if instance.get('num_meds', 0) > 0)
    num_with_addiction = count_categorical(data, 'addiction', 'y')

    num_nic_other = count_categorical(data, 'nic_other', 'y')
    num_cannabis = count_categorical(data, 'cannabis', 'y')
    num_opioids = count_categorical(data, 'opioids', 'y')
    num_other_drugs = count_categorical(data, 'other_drugs', 'y')

    occupation_danger = count_distribution(data, 'occup_danger')
    lifestyle_danger = count_distribution(data, 'ls_danger')

    diseases = {
        "diabetes": count_categorical(data, 'diabetes', 'y'),
        "heart_disease_or_stroke": count_categorical(data, 'hds', 'y'),
        "asthma": count_categorical(data, 'asthma', 'y'),
        "immune_deficiency": count_categorical(data, 'immune_defic', 'y')
    }

    family_history = {
        "family_cancer": count_categorical(data, 'family_cancer', 'y'),
        "family_heart_disease": count_categorical(data, 'family_heart_disease', 'y'),
        "family_cholesterol": count_categorical(data, 'family_cholesterol', 'y')
    }

    # Stampare i risultati
    print(f"Numero di istanze: {num_instances}")
    print(f"Età media: {avg_age:.2f}")
    print(f"Peso medio: {avg_weight:.2f} lb")
    print(f"Numero di uomini: {num_men}")
    print(f"Numero di donne: {num_women}")
    print(f"Altezza media: {avg_height:.2f} pollici")
    print(f"Pressione sanguigna media (sistolica): {avg_sys_bp:.2f} mmHg")
    print(f"Numero di fumatori: {num_smokers}")
    print(f"Numero di persone che fanno uso di altri tipi di nicotina: {num_nic_other}")
    print(f"Numero di persone che assumono medicinali: {num_people_on_meds}")
    print(f"Distribuzione delle occupazioni per livello di pericolo: {occupation_danger}")
    print(f"Distribuzione del rischio dello stile di vita: {lifestyle_danger}")
    print(f"Numero di persone che assumano cannabis: {num_cannabis}")
    print(f"Numero di persone che assumano oppioidi: {num_opioids}")
    print(f"Numero di persone che assumano altre droghe: {num_other_drugs}")
    print(f"Numero medio di drink a settimana: {avg_drinks:.2f}")
    print(f"Numero di persone con dipendenze: {num_with_addiction}")
    print(f"Numero di persone che hanno subito interventi chirurgici: {avg_surgeries:.2f}")
    print(f"Colesterolo medio: {avg_cholesterol:.2f}")
    print(f"Malattie riscontrate: {diseases}")
    print(f"Storia familiare di malattie: {family_history}")

    # Analisi della distribuzione dell'età alla morte
    analyze_target_distribution(data, target='age')

    # Calcola la media dell'età di morte per fumatori e non fumatori
    avg_death_age_smokers, avg_death_age_non_smokers = calculate_smoking_groups_average(data)
    if avg_death_age_smokers is not None and avg_death_age_non_smokers is not None:
        print(f"\nEtà media di morte per i fumatori: {avg_death_age_smokers:.2f}")
        print(f"Età media di morte per i non fumatori: {avg_death_age_non_smokers:.2f}")

if __name__ == '__main__':
    main()
