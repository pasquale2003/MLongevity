import json
import os
from collections import Counter
import math


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

def load_data(file_path):
    """Carica il file JSON e restituisce i dati."""
    with open(file_path, 'r') as file:
        return json.load(file)


def calculate_average(data, key):
    """Calcola la media di un dato numerico specifico."""
    values = [instance.get(key) for instance in data if isinstance(instance.get(key), (int, float))]
    return sum(values) / len(values) if values else 0


def calculate_std_dev(data, key, mean):
    """Calcola la deviazione standard di un dato numerico specifico."""
    values = [instance.get(key) for instance in data if isinstance(instance.get(key), (int, float))]
    variance = sum((x - mean) ** 2 for x in values) / len(values) if values else 0
    return math.sqrt(variance)


def categorize_age(data, mean, std_dev):
    """Categorizza le persone in base all'età rispetto alla media e alla deviazione standard."""
    categories = {
        "meno della media": 0,
        "nella media": 0,
        "più della media": 0
    }

    for instance in data:
        age = instance.get("age")
        if isinstance(age, (int, float)):
            if age < (mean - std_dev):
                categories["meno della media"] += 1
            elif age > (mean + std_dev):
                categories["più della media"] += 1
            else:
                categories["nella media"] += 1
    return categories

def main():
    # Caricare i dati dal file JSON
    data = load_data('../data/data.json')

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

    # Calcolo media e deviazione standard dell'età
    avg_age = calculate_average(data, 'age')
    std_dev_age = calculate_std_dev(data, 'age', avg_age)

    # Categorizzazione delle età
    age_categories = categorize_age(data, avg_age, std_dev_age)

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

    # Stampare i risultati
    print(f"\nEtà media: {avg_age:.2f}")
    print(f"Deviazione standard dell'età: {std_dev_age:.2f}")
    print(f"Distribuzione delle categorie di età: {age_categories}")

if __name__ == '__main__':
    main()
