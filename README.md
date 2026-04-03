# Empiryczna analiza złożoności mnożenia macierzy (Matrix Multiplication Complexity) 

## Opis projektu
Projekt stanowi rozwiązanie zadania akademickiego dotyczącego empirycznej analizy czasu mnożenia gęstych macierzy z wykorzystaniem biblioteki NumPy.

Celem projektu jest przeprowadzenie eksperymentu obliczeniowego dla różnych rozmiarów macierzy, zebranie danych pomiarowych oraz dopasowanie i porównanie modeli regresyjnych opisujących zależność czasu obliczeń od rozmiaru problemu.

## Zakres eksperymentu
W eksperymencie wykonywano mnożenie macierzy w postaci:

A(p × q) × B(q × r)

z użyciem operatora `A @ B` w bibliotece NumPy.

Dla wielu kombinacji wymiarów `p`, `q` i `r` mierzono czas wykonania operacji. Każdy punkt pomiarowy był powtarzany wielokrotnie, a w danych zapisywano podstawowe statystyki czasu obliczeń.

## Zebrane dane
Każdy rekord danych zawiera następujące pola:
- `p`
- `q`
- `r`
- `time_median`
- `time_mean`
- `time_std`

W analizie można również uwzględniać zmienne pochodne, na przykład iloczyn `pqr`, który opisuje skalę problemu obliczeniowego.

## Modelowanie
W projekcie dopasowano co najmniej trzy modele regresyjne opisujące zależność czasu wykonania od rozmiaru problemu.

Uwzględniono między innymi:
- model log-liniowy:
  `log(T) = α + βp log(p) + βq log(q) + βr log(r) + ε`
- model logarytmiczny z jedną zmienną złożonościową:
  `log(T) = α + β log(pqr) + ε`
- model liniowy w danych surowych:
  `T = α + γ(pqr) + ε`

Celem porównania modeli było sprawdzenie, który z nich najlepiej opisuje zależność empirycznego czasu wykonania od rozmiaru mnożonych macierzy.

## Zawartość repozytorium
- `README.md` — opis projektu,
- `matrix_multiplication_complexity.py` lub `matrix_multiplication_complexity.ipynb` — kod eksperymentu i analizy,
- `matrix_multiplication_data.csv` — dane eksperymentalne,
- `raport-empiryczna-analiza-zlozonosci-mnozenia-macierzy.pdf` — raport końcowy.

## Wykorzystane narzędzia
Projekt został przygotowany w Pythonie z użyciem bibliotek:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `statsmodels` lub innych narzędzi do regresji

## Wyniki analizy
W projekcie analizowano:
- jak czas mnożenia zmienia się wraz z rozmiarem macierzy,
- które zmienne najlepiej opisują koszt obliczeniowy,
- czy zależność czasu od rozmiaru problemu jest dobrze opisywana przez modele liniowe i log-liniowe,
- który model regresyjny najlepiej dopasowuje się do danych eksperymentalnych.

## Wnioski
Projekt pozwala porównać obserwacje empiryczne z intuicją teoretyczną dotyczącą złożoności mnożenia macierzy.

Analiza pokazuje, że czas wykonania zależy silnie od wymiarów `p`, `q` i `r`, a odpowiednio dobrane modele regresyjne pozwalają opisać tę zależność i porównać jakość różnych przybliżeń.

## Autor
Projekt wykonany w ramach zadania akademickiego z modeli analizy danych.

## License
This project is provided for viewing, downloading, running, and private modification only.
Redistribution, republication, commercial use, and claiming authorship or ownership are prohibited without prior written permission from the author.
