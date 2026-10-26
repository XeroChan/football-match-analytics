# Football Match Analytics ⚽📊

Projekt przeznaczony do eksploracji, czyszczenia i analizy danych pomeczowych w formacie JSON (dane taktyczno-zdarzeniowe).

---

## 📂 1. Struktura katalogów projektu

```text
football_analytics/
├── data/
│   ├── raw/                  # Surowe pliki .json (np. match_1821123...json)
│   └── processed/            # Przetworzone, spłaszczone tabele (format .parquet)
├── notebooks/
│   └── 01_eda_events.ipynb   # Notatniki Jupyter do szybkiej eksploracji i wykresów
├── src/
│   └── parser.py             # Funkcje do czyszczenia danych (np. rozpakowywanie 'qualifiers')
├── .venv/                    # Wirtualne środowisko Pythona
└── requirements.txt          # Lista bibliotek

```

---

## 🛠️ 2. Konfiguracja środowiska w VS Code

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

### Rekomendowane rozszerzenia (Extensions) do VS Code

* **Python** oraz **Jupyter** (Microsoft) – obsługa i podpowiadanie kodu w notatnikach `.ipynb`.
* **Data Wrangler** (Microsoft) – interaktywny podgląd i analiza ramek danych (DataFrame) bez pisania kodu.
* **Rainbow CSV / Parquet Viewer** – wygodne przeglądanie spodystkowanych plików wynikowych.

---

## 🗄️ 3. Architektura danych (Schemat tabel)

| Tabela docelowa | Źródło w pliku JSON | Kluczowe informacje do wyciągnięcia |
| --- | --- | --- |
| **`matches_teams`** | Główne klucze pliku, `away` / `home` | Frekwencja (`attendance`), wynik (`scores`), menedżer (`managerName`), średni wiek (`averageAge`), ID oraz nazwa drużyny (`teamId`, `name`). |
| **`players`** | `away.players` / `home.players` | Dane demograficzne (`age`, `height`, `weight`), pozycja (`position`), wyjściowa jedenastka (`isFirstEleven`), czas zmian (`subbedInExpandedMinute` / `subbedOutExpandedMinute`). |
| **`events`** | Główna tablica `events` oraz `incidentEvents` | Oś czasu (`minute`, `second`), współrzędne boiska (`x`, `y`, `endX`, `endY`), typ akcji (`type.displayName`), rezultat (`outcomeType.displayName`). |
| **`formations`** | `away.formations` / `home.formations` | Zmiany taktyczne (`formationName`, `startMinuteExpanded`, `endMinuteExpanded`) oraz pozycje zawodników (`formationPositions`). |

---

## 🔍 4. Główne kierunki eksploracji (EDA)

1. **Analiza przestrzenna (Spatial Analysis):** Wykorzystanie współrzędnych `x`, `y` (skala 0–100) z tabeli `events`.

* **Mapy podań (Pass Networks):** Wyznaczanie sieci wymiany podań (`Pass` o statusie `Successful`) i średnich pozycji zawodników.
* **Mapy strzałów (Shot Maps):** Filtrowanie zdarzeń z flagą `isShot: true` z uwzględnieniem współrzędnych bramki (`goalMouthY`, `goalMouthZ`) oraz cech takich jak `BigChance` czy `Penalty`.

1. **Wpływ zmian i taktyki:** Badanie momentów roszad w składzie (np. wejście rezerwowych w `expandedMinute: 67`) oraz wpływu zmiany formacji na zagęszczenie akcji w tercji ataku (`finalThird`).
2. **Profilowanie zawodników:** Korelacje parametrów fizycznych (`height`, `weight`) ze skutecznością w powietrzu (`aerialsWon`) oraz identyfikacja podań progresywnych na podstawie kąta i długości.

---

## 💡 5. Codzienny Workflow pracy

1. Stwórz lub otwórz plik roboczy w katalogu `notebooks/`.
2. W prawym górnym rogu notatnika w VS Code wybierz **Select Kernel** i wskaż środowisko `.venv`.
3. Przetestuj logikę na 1–2 meczach w notatniku.
4. Gotowe, uniwersalne funkcje czyszczące przenoś do modułu w `src/parser.py`, aby utrzymać porządek w kodzie.
