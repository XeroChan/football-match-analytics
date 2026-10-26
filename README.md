# Football Match Analytics ⚽📊

Projekt przeznaczony do pobierania, czyszczenia i analizy danych pomeczowych WhoScored (`matchCentreData` JSON) w tabelarycznych formatach **Parquet** oraz **CSV**.

---

## 📂 1. Struktura katalogów projektu

```text
football-match-analytics/
├── data/
│   ├── raw/                  # Surowe pliki .json (np. match_1821123_..._raw.json)
│   └── processed/            # Przetworzone tabele w formacie .parquet (szybka analiza wielu meczów)
├── output/                   # Przetworzone tabele w formacie .csv (podgląd w Excelu / Data Wranglerze)
├── notebooks/
│   └── 01_eda_events.ipynb   # Notatnik Jupyter korzystający z modułu src.parser
├── src/
│   ├── __init__.py           # Eksport głównych klas pakietu
│   ├── main.py               # Punkt wejścia CLI i orkiestrator potoku danych
│   ├── scraper.py            # Pobieranie danych matchCentreData przez Selenium
│   ├── parser.py             # Transformacja JSON -> DataFrame (events, players, formations, metadata, stats...)
│   └── exporter.py           # Zapis surowego JSON-a oraz tabel Parquet i CSV
└── requirements.txt          # Lista bibliotek Pythona
```

---

## 🛠️ 2. Konfiguracja i uruchomienie

### Tworzenie środowiska i instalacja bibliotek

```bash
# 1. Utworzenie wirtualnego środowiska
python -m venv .venv

# 2. Aktywacja środowiska (PowerShell / Windows)
.\.venv\Scripts\Activate.ps1
# (Lub w systemach macOS/Linux: source .venv/bin/activate)

# 3. Instalacja wymaganych pakietów
pip install -r requirements.txt
```

### Uruchamianie potoku danych (CLI)

```bash
# 1. Przetworzenie domyślnego meczu (automatycznie użyje pliku z data/raw/, jeśli już istnieje)
python -m src.main

# 2. Pobranie jednego lub wielu meczów z linków WhoScored Live
python -m src.main --url "https://www.whoscored.com/Matches/1821123/Live/England-Premier-League"

# 3. Wymuszenie ponownego pobrania przez przeglądarkę (pominięcie cache'u JSON)
python -m src.main --url "https://www.whoscored.com/Matches/1821123/Live/England-Premier-League" --force-scrape

# 4. Przetworzenie lokalnych plików JSON offline (bez uruchamiania Chrome)
python -m src.main --from-json "data/raw/match_1821123_Southampton_vs_Aston_Villa_raw.json"
```

---

## 🗄️ 3. Architektura danych (Schemat tabel)

`MatchDataParser.parse_all()` zwraca obiekt `ParsedMatchDataset` zawierający 10 ustrukturyzowanych tabel:

| Tabela (`DataFrame`) | Źródło w pliku JSON | Kluczowe informacje |
| --- | --- | --- |
| **`metadata_df`** | Główne klucze pliku, `home`, `away`, `referee` | Wynik (`score`, `ht_score`, `ft_score`), frekwencja (`attendance`), stadion (`venue_name`), sędzia (`referee_name`), menedżerowie, wyjściowe formacje, średni wiek drużyn, wyniki drużyn (`home/away_score_*`) oraz granice połów (`period_N_minute_limit`, `period_N_end_minute`). |
| **`players_df`** | `home.players` / `away.players` | Składy (`is_first_eleven`), pozycja (`position`), numer (`shirt_no`), dane fizyczne (`age`, `height_cm`, `weight_kg`), minuty i połowy zmian (`subbed_in_period`, `subbed_out_period`) oraz ocena końcowa (`final_rating`). |
| **`events_df`** | Główna tablica `events` + słownik graczy | Oś czasu (`minute`, `second`, `expanded_minute`), współrzędne (`x`, `y`, `end_x`, `end_y`), typ i wynik akcji, spłaszczone kwalifikatory (`is_key_pass`, `is_big_chance`, `pass_length`, `pass_angle`) zdarzenia powiązane (`related_event_id`, `related_player_id`), opis incydentów (`event_text`, `satisfiers`), rozkodowane `satisfied_event_type_names` oraz wywnioskowany odbiorca podania (`receiver_id`, `receiver_name`). |
| **`formations_df`** | `home.formations` / `away.formations` | Przedziały czasowe ustawień taktycznych (`formation_name`, `start_minute_expanded`, `end_minute_expanded`) wraz z przypisanymi zawodnikami i ich współrzędnymi (`horizontal`, `vertical`). |
| **`player_stats_df`** | `home/away.players[].stats` | Statystyki zawodników minuta po minucie w formacie long (`stat_name`, `minute`, `value`), m.in. pełna seria `ratings`, `passesTotal`, `touches`, `totalSaves`. |
| **`team_stats_df`** | `home.stats` / `away.stats` | 34 metryki drużynowe minuta po minucie w formacie long (posiadanie, strzały, odbiory, pojedynki powietrzne, oceny). |
| **`shot_zones_df`** | `home/away.shotZones` | Liczba strzałów i goli (`count`, `goal_count`) w strefach bramki (`zone`) w danej minucie. |
| **`minute_mapping_df`** | `expandedMinutes` | Mapowanie (`period_value`, `minute`) → `expanded_minute`. |
| **`lookups_df`** | `_matchCentreEventTypeJson`, `_formationIdNameMappings` | Słowniki ID → nazwa (`lookup_type`: `event_type` / `formation`). |
| **`common_events_df`** | `commonEvents` | Ogólne spłaszczenie elementów listy (`pd.json_normalize`) + surowy JSON w kolumnie `raw_json`. Tabela powstaje tylko, gdy lista nie jest pusta. |

> Pomijane celowo: `incidentEvents` (duplikaty `events`), `stats.minutesWithStats` (sam indeks minut) oraz techniczne pola `$idx` / `$len`.

---

## 🔍 4. Główne kierunki eksploracji (EDA)

1. **Analiza przestrzenna (Spatial Analysis):** Wykorzystanie współrzędnych `x`, `y` (skala 0–100) z tabeli `events_df`.
   * **Mapy podań (Pass Networks):** Filtrowanie celnych podań (`event_type == 'Pass'` oraz `outcome_type == 'Successful'`) z gotowymi kolumnami `player_name` -> `receiver_name`.
   * **Mapy strzałów (Shot Maps):** Filtrowanie zdarzeń z flagą `is_shot == True` z uwzględnieniem współrzędnych bramki (`goal_mouth_y`, `goal_mouth_z`) oraz flag `is_big_chance` i `is_penalty`.
2. **Wpływ zmian i taktyki:** Analiza przedziałów w `formations_df` (np. przejście na inne ustawienie w `start_minute_expanded: 67`) oraz wpływu zmian na wydarzenia boiskowe.
3. **Skalowalna analiza wielu meczów (DuckDB / Polars):** Dzięki zapisowi do `data/processed/*.parquet` można odpytywać setki meczów jednym zapytaniem SQL przez DuckDB lub Polars bez wczytywania plików CSV w pętli.
