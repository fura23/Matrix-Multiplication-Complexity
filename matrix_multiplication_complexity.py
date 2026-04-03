import numpy as np
import pandas as pd
import time
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# --- 1. Konfiguracja eksperymentu ---
# Zakresy wymiarów macierzy
sizes = [100, 200, 300, 400, 500, 750, 1000, 1250, 1500]
repeats = 5  # Liczba powtórzeń dla każdego punktu pomiarowego

results = []

print("Rozpoczynanie eksperymentu analizy mnożenia macierzy...")

# --- 2. Zbieranie danych ---
scenarios = []
# Macierze kwadratowe (p=q=r)
for s in sizes:
    scenarios.append((s, s, s))
# Macierze prostokątne
fixed = 500
for s in sizes:
    if s != fixed:
        scenarios.append((s, fixed, fixed)) 
        scenarios.append((fixed, s, fixed)) 
        scenarios.append((fixed, fixed, s)) 

print(f"Liczba scenariuszy do przebadania: {len(scenarios)}")

for i, (p, q, r) in enumerate(scenarios):
    timings = []
    
    # Generowanie losowych macierzy gęstych
    A = np.random.rand(p, q)
    B = np.random.rand(q, r)
    
    # Rozgrzewka (warm-up) dla cache procesora
    _ = A @ B
    
    # Pętla pomiarowa
    for _ in range(repeats):
        start_time = time.perf_counter()
        _ = A @ B
        end_time = time.perf_counter()
        timings.append(end_time - start_time)
    
    # Obliczanie statystyk
    median_time = np.median(timings)
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    
    results.append({
        'p': p, 'q': q, 'r': r,
        'time_median': median_time,
        'time_mean': mean_time,
        'time_std': std_time,
        'pqr': p * q * r
    })
    
    print(f"[{i+1}/{len(scenarios)}] {p}x{q}x{r} -> {median_time:.4f}s")

# Tworzenie DataFrame
df = pd.DataFrame(results)

# Transformacje logarytmiczne potrzebne do modeli regresji
df['log_p'] = np.log(df['p'])
df['log_q'] = np.log(df['q'])
df['log_r'] = np.log(df['r'])
df['log_time'] = np.log(df['time_median'])
df['log_pqr'] = np.log(df['pqr'])

# Zapis danych do pliku CSV
csv_filename = 'matrix_multiplication_data.csv'
df.to_csv(csv_filename, index=False)
print(f"\nDane zapisano do pliku '{csv_filename}'.")

# --- 3. Modelowanie regresyjne ---

print("\n--- Wyniki regresji ---")

# Model 1: Log-liniowy pełny
# log(T) = alpha + beta_p*log(p) + beta_q*log(q) + beta_r*log(r)
model1 = smf.ols('log_time ~ log_p + log_q + log_r', data=df).fit()
print("\nMODEL 1 (Log-liniowy pełny):")
print(f"R-squared: {model1.rsquared:.4f}")
print(model1.params)

# Model 2: Logarytmiczny (zmienna pqr)
# log(T) = alpha + beta*log(pqr)
model2 = smf.ols('log_time ~ log_pqr', data=df).fit()
print("\nMODEL 2 (Logarytmiczny pqr):")
print(f"R-squared: {model2.rsquared:.4f}")
print(model2.params)

# Model 3: Liniowy (dane surowe)
# T = alpha + gamma*(pqr)
model3 = smf.ols('time_median ~ pqr', data=df).fit()
print("\nMODEL 3 (Liniowy surowy):")
print(f"R-squared: {model3.rsquared:.4f}")
print(model3.params)

# --- 4. Generowanie wykresów ---

print("\nGenerowanie wykresów...")

# Wykres 1: Model 1 (Przewidywane vs Obserwowane)
plt.figure(figsize=(8, 6))
predicted = model1.predict(df)
plt.scatter(df['log_time'], predicted, color='blue', alpha=0.6, label='Dane pomiarowe')
min_val = min(df['log_time'].min(), predicted.min())
max_val = max(df['log_time'].max(), predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Idealne dopasowanie')
plt.xlabel("Logarytm czasu obserwowanego")
plt.ylabel("Logarytm czasu przewidywanego")
plt.title(f"Model 1: Dopasowanie (R2={model1.rsquared:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wykres_1_model_log_liniowy.png')
plt.close()

# Wykres 2: Model 2 (Skala logarytmiczna)
plt.figure(figsize=(8, 6))
plt.scatter(df['log_pqr'], df['log_time'], color='green', alpha=0.6, label='Dane pomiarowe')
x_vals = np.linspace(df['log_pqr'].min(), df['log_pqr'].max(), 100)
y_vals = model2.params['Intercept'] + model2.params['log_pqr'] * x_vals
plt.plot(x_vals, y_vals, 'r--', lw=2, label='Linia regresji')
plt.xlabel("Logarytm złożoności log(p*q*r)")
plt.ylabel("Logarytm czasu log(t)")
plt.title(f"Model 2: Skala logarytmiczna (R2={model2.rsquared:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wykres_2_model_log_pqr.png')
plt.close()

# Wykres 3: Model 3 (Skala liniowa)
plt.figure(figsize=(8, 6))
plt.scatter(df['pqr'], df['time_median'], color='purple', alpha=0.6, label='Dane pomiarowe')
x_vals_lin = np.linspace(df['pqr'].min(), df['pqr'].max(), 100)
y_vals_lin = model3.params['Intercept'] + model3.params['pqr'] * x_vals_lin
plt.plot(x_vals_lin, y_vals_lin, 'r--', lw=2, label='Linia regresji')
plt.xlabel("Złożoność (p*q*r)")
plt.ylabel("Czas wykonania (s)")
plt.title(f"Model 3: Skala liniowa (R2={model3.rsquared:.3f})")
# Notacja naukowa na osi X dla czytelności
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('wykres_3_model_liniowy.png')
plt.close()

