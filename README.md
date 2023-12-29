# geo_price


### Struktura plików:
```
├── scripts
    ├── utils.py # załadowanie danych
    ├── prepare.py # czyszczenie kolumn i wekoryzacja
    ├── dataset.py # utworzenie datasetu
    ├── autoencoder.py # implementacja autoenkodera
    └── train.py # tranowanie autoenkodera
└── data
    ├── prices # folder z danymi o cenach mieszkań
    └── demographic # folder z danymi demograficznymi
```

### TO DO (dla części bez danych przestrzennych):
- Dodanie requirements.txt
- Wytrenowanie autoenkodera
- Regresja za pomocą KNN
- Wizaualizacja / Ewalizacja wyników

### Pozyskanie danych:
```
dvc pull
```