# geo_price


### Struktura plików:
```
├── scripts
    ├── utils.py # załadowanie danych
    ├── prepare.py # czyszczenie kolumn i wekoryzacja
    ├── dataset.py # utworzenie datasetu
    ├── autoencoder.py # implementacja autoenkodera
    ├── encode.py # tranowanie autoenkodera
    └── train.py # tranowanie regresji za pomocą KNN
└── data
    ├── prices # folder z danymi o cenach mieszkań
    └── demographic # folder z danymi demograficznymi
```

### TO DO
- [ ] demo

### Pozyskanie danych:
```
dvc pull
```
Po zalogowaniu w przeglądarce plik autoryzacyjny zapisze się w lokalizacji `../mycredentials.json`

### Uruchomienie skryptu:
```
python run.py 
``` 
W kodzie można zdefiniować jaką metodę chcemy uruchomić.

### Wymagania:
```
pip install -r requirements.txt
```

### Demo:
W pliku `demo.ipynb` znajduje się podsumowanie wyników z wizualizacją wyników, w postaci wykresów i interaktywnych map.
