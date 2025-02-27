import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# Carica il dataset di test
def load_test_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Codifica variabili categoriche
def encode_features(data):
    le = LabelEncoder()
    categorical_columns = ['sex', 'smoker', 'nic_other', 'cannabis', 'opioids', 'other_drugs', 'addiction',
                           'diabetes', 'hds', 'asthma', 'immune_defic', 'family_cancer', 'family_heart_disease',
                           'family_cholesterol']
    for col in categorical_columns:
        for instance in data:
            instance[col] = le.fit_transform([instance[col]])[0]
    return data

# Previsione
def predict(model, data):
    X = np.array([list(instance.values())[:-1] for instance in data])
    y_true = np.array([instance['age'] for instance in data])
    y_pred = model.predict(X)
    return y_true, y_pred

# Allena un modello di esempio
def train_example_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# Calcola le metriche
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("=== Metriche del Modello ===")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print("===========================")

# Validazione incrociata
def cross_validation(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Valutazione Cross-Validation (MSE): {cv_scores}")
    print(f"MSE Medio della Cross-Validation: {-cv_scores.mean():.4f}")

# Grafico dei Residui
def plot_residuals(y_true, y_pred):
    residui = y_true - y_pred
    plt.scatter(y_pred, residui, color="blue", alpha=0.5)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Valori Predetti")
    plt.ylabel("Residui")
    plt.title("Grafico dei Residui")
    plt.show()

# Grafico Valori Reali vs Predetti
def plot_real_vs_pred(y_true, y_pred):
    plt.scatter(y_true, y_pred, color="blue", alpha=0.5, label="Predizioni")
    plt.plot(y_true, y_true, color="red", linestyle="--", label="Perfetta Corrispondenza")
    plt.xlabel("Valori Reali")
    plt.ylabel("Valori Predetti")
    plt.title("Valori Reali vs Predetti")
    plt.legend()
    plt.show()

# Curva di apprendimento
def plot_learning_curve(model, X_train, y_train, X_test, y_test):
    train_errors, test_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        train_errors.append(mean_squared_error(y_train[:m], model.predict(X_train[:m])))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test)))

    plt.plot(np.sqrt(train_errors), label="Training Error")
    plt.plot(np.sqrt(test_errors), label="Test Error")
    plt.legend()
    plt.title("Curva di Apprendimento")
    plt.xlabel("Numero di Esempi di Addestramento")
    plt.ylabel("Errore (RMSE)")
    plt.show()

def main():
    # Carica il file di test
    data = load_test_data('../data/data_test.json')

    # Codifica le feature categoriche
    data = encode_features(data)

    # Prepara i dati
    X = np.array([list(instance.values())[:-1] for instance in data])
    y = np.array([instance['age'] for instance in data])

    # Dividi i dati
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Allena il modello
    model = train_example_model(X_train, y_train)

    # Previsione e metriche
    y_true, y_pred = predict(model, data)
    calculate_metrics(y_true, y_pred)

    # Valutazione con cross-validation
    cross_validation(model, X, y)

    # Generazione grafici
    plot_residuals(y_true, y_pred)
    plot_real_vs_pred(y_true, y_pred)
    plot_learning_curve(model, X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()
