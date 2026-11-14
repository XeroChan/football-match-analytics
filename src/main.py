"""
main.py
=======
Main CLI and pipeline orchestrator for scraping WhoScored Match Centre data
and exporting structured datasets (`events`, `players`, `formations`, `metadata`,
`player_stats`, `team_stats`, `shot_zones`, `minute_mapping`, `lookups`, `common_events`)
in Parquet (`data/processed/`) and CSV (`output/`) formats alongside raw JSON backups,
plus combined multi-match datasets in `output/combined/` and `data/processed/combined/`.

Usage examples:
    1. Scrape (or load from local cache) the configured array of match URLs (`DEFAULT_MATCH_URLS`):
       python -m src.main

    2. Scrape one or more custom WhoScored Live match URLs:
       python -m src.main --url "https://www.whoscored.com/Matches/1821123/Live/..." "https://..."

    3. Re-generate Parquet, per-match CSV, and combined CSV files offline from saved raw JSON backups:
       python -m src.main --from-json data/raw/*_raw.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

from src.exporter import ExportedFiles, MatchDataExporter
from src.models import ParsedMatchDataset
from src.parser import MatchDataParser
from src.scraper import WhoScoredScraper, WhoScoredScraperError

logger = logging.getLogger(__name__)

# Array of WhoScored Live match URLs to scrape/process by default when running `python -m src.main`.
DEFAULT_MATCH_URLS: list[str] = [
    "https://www.whoscored.com/matches/1821123/live/england-premier-league-2024-2025-southampton-aston_villa",
    "https://www.whoscored.com/matches/1983593/live/england-premier-league-2026-2027-chelsea-bournemouth",
    "https://www.whoscored.com/matches/1983603/live/england-premier-league-2026-2027-liverpool-manchester-city",
    "https://www.whoscored.com/matches/1993972/live/spain-laliga-2026-2027-barcelona-deportivo-alaves",
    "https://www.whoscored.com/matches/1993980/live/spain-laliga-2026-2027-levante-atletico-madrid",
    "https://www.whoscored.com/matches/1993983/live/spain-laliga-2026-2027-racing-santander-real-madrid",
    "https://www.whoscored.com/matches/1995426/live/germany-bundesliga-2026-2027-werder-bremen-paderborn",
    "https://www.whoscored.com/matches/1998572/live/international-fifa-world-cup-2026-switzerland-colombia",
    "https://www.whoscored.com/matches/2002449/live/international-fifa-world-cup-2026-france-spain",
    "https://www.whoscored.com/matches/2029242/live/europe-champions-league-2026-2027-viking-bayern-munich",
    "https://www.whoscored.com/matches/1991112/live/russia-premier-league-2026-2027-krylya-sovetov-samara-fc-orenburg",
    "https://www.whoscored.com/matches/1991103/live/russia-premier-league-2026-2027-rubin-kazan-fakel",
    "https://www.whoscored.com/matches/1999627/live/turkey-super-lig-2026-2027-eyupspor-kocaelispor",
    "https://www.whoscored.com/matches/1985905/live/belgium-jupiler-pro-league-2026-2027-kortrijk-kv-mechelen",
    "https://www.whoscored.com/matches/1995758/live/germany-2-bundesliga-2026-2027-arminia-bielefeld-eintracht-braunschweig",
    "https://www.whoscored.com/matches/1995450/live/germany-bundesliga-2026-2027-bayern-munich-borussia-dortmund",
    "https://www.whoscored.com/matches/1995458/live/germany-bundesliga-2026-2027-union-berlin-fc-koln",
    "https://www.whoscored.com/matches/1995418/live/germany-bundesliga-2026-2027-elversberg-augsburg",
    "https://www.whoscored.com/matches/1995411/live/germany-bundesliga-2026-2027-bayern-munich-rb-leipzig",
    "https://www.whoscored.com/matches/1995424/live/germany-bundesliga-2026-2027-union-berlin-borussia-dortmund",
    "https://www.whoscored.com/matches/1993932/live/spain-laliga-2026-2027-celta-vigo-real-betis",
    "https://www.whoscored.com/matches/1993934/live/spain-laliga-2026-2027-barcelona-real-madrid",
    "https://www.whoscored.com/matches/1994185/live/spain-laliga-2026-2027-getafe-rayo-vallecano",
    "https://www.whoscored.com/matches/1994189/live/spain-laliga-2026-2027-valencia-athletic-club",
    "https://www.whoscored.com/matches/1984218/live/france-ligue-1-2026-2027-le-havre-paris-saint-germain",
    "https://www.whoscored.com/matches/1984096/live/france-ligue-1-2026-2027-paris-fc-monaco",
    "https://www.whoscored.com/matches/1984184/live/france-ligue-1-2026-2027-auxerre-le-mans",
    "https://www.whoscored.com/matches/1984143/live/france-ligue-1-2026-2027-angers-marseille",
    "https://www.whoscored.com/matches/1956076/live/brazil-brasileirão-2026-vitoria-athletico-paranaense",
    "https://www.whoscored.com/matches/1956089/live/brazil-brasileirão-2026-bahia-sao-paulo",
    "https://www.whoscored.com/matches/1956054/live/brazil-brasileirão-2026-gremio-cruzeiro",
    "https://www.whoscored.com/matches/1956062/live/brazil-brasileirão-2026-vasco-da-gama-corinthians",
    "https://www.whoscored.com/matches/1953075/live/usa-major-league-soccer-2026-inter-miami-cf-fc-cincinnati",
    "https://www.whoscored.com/matches/1953076/live/usa-major-league-soccer-2026-new-england-revolution-red-bull-new-york",
    "https://www.whoscored.com/matches/1953095/live/usa-major-league-soccer-2026-la-galaxy-austin-fc",
    "https://www.whoscored.com/matches/1953064/live/usa-major-league-soccer-2026-orlando-city-new-york-city-fc",
    "https://www.whoscored.com/matches/1953094/live/usa-major-league-soccer-2026-sporting-kansas-city-fc-dallas",
    "https://www.whoscored.com/matches/1983594/live/england-premier-league-2026-2027-arsenal-tottenham",
    "https://www.whoscored.com/matches/1983595/live/england-premier-league-2026-2027-manchester-united-newcastle-united",
    "https://www.whoscored.com/matches/1983596/live/england-premier-league-2026-2027-aston-villa-west-ham",
    "https://www.whoscored.com/matches/1983597/live/england-premier-league-2026-2027-brighton-everton",
    "https://www.whoscored.com/matches/1983598/live/england-premier-league-2026-2027-fulham-crystal-palace",
    "https://www.whoscored.com/matches/1983599/live/england-premier-league-2026-2027-brentford-wolverhampton-wanderers",
    "https://www.whoscored.com/matches/1983600/live/england-premier-league-2026-2027-nottingham-forest-leicester",
    "https://www.whoscored.com/matches/1983601/live/england-premier-league-2026-2027-leeds-burnley",
    "https://www.whoscored.com/matches/1983602/live/england-premier-league-2026-2027-sunderland-sheffield-united",
    "https://www.whoscored.com/matches/1983604/live/england-premier-league-2026-2027-manchester-city-arsenal",
    "https://www.whoscored.com/matches/1983605/live/england-premier-league-2026-2027-tottenham-chelsea",
    "https://www.whoscored.com/matches/1983606/live/england-premier-league-2026-2027-newcastle-united-liverpool",
    "https://www.whoscored.com/matches/1983607/live/england-premier-league-2026-2027-west-ham-manchester-united",
    "https://www.whoscored.com/matches/1983608/live/england-premier-league-2026-2027-everton-aston-villa",
    "https://www.whoscored.com/matches/1983609/live/england-premier-league-2026-2027-crystal-palace-brighton",
    "https://www.whoscored.com/matches/1983610/live/england-premier-league-2026-2027-wolverhampton-wanderers-fulham",
    "https://www.whoscored.com/matches/1983611/live/england-premier-league-2026-2027-leicester-brentford",
    "https://www.whoscored.com/matches/1983612/live/england-premier-league-2026-2027-bournemouth-nottingham-forest",
    "https://www.whoscored.com/matches/1983613/live/england-premier-league-2026-2027-burnley-sunderland",
    "https://www.whoscored.com/matches/1983614/live/england-premier-league-2026-2027-sheffield-united-leeds",
    "https://www.whoscored.com/matches/1983615/live/england-premier-league-2026-2027-liverpool-arsenal",
    "https://www.whoscored.com/matches/1983616/live/england-premier-league-2026-2027-chelsea-manchester-united",
    "https://www.whoscored.com/matches/1983617/live/england-premier-league-2026-2027-manchester-city-tottenham",
    "https://www.whoscored.com/matches/1983618/live/england-premier-league-2026-2027-aston-villa-newcastle-united",
    "https://www.whoscored.com/matches/1983619/live/england-premier-league-2026-2027-brighton-west-ham",
    "https://www.whoscored.com/matches/1983620/live/england-premier-league-2026-2027-fulham-everton",
    "https://www.whoscored.com/matches/1983621/live/england-premier-league-2026-2027-brentford-crystal-palace",
    "https://www.whoscored.com/matches/1983622/live/england-premier-league-2026-2027-nottingham-forest-wolverhampton-wanderers",
    "https://www.whoscored.com/matches/1983623/live/england-premier-league-2026-2027-leeds-leicester",
    "https://www.whoscored.com/matches/1983624/live/england-premier-league-2026-2027-sunderland-bournemouth",
    "https://www.whoscored.com/matches/1983625/live/england-premier-league-2026-2027-arsenal-chelsea",
    "https://www.whoscored.com/matches/1983626/live/england-premier-league-2026-2027-manchester-united-liverpool",
    "https://www.whoscored.com/matches/1983627/live/england-premier-league-2026-2027-tottenham-aston-villa",
    "https://www.whoscored.com/matches/1983628/live/england-premier-league-2026-2027-newcastle-united-manchester-city",
    "https://www.whoscored.com/matches/1983629/live/england-premier-league-2026-2027-west-ham-fulham",
    "https://www.whoscored.com/matches/1983630/live/england-premier-league-2026-2027-everton-brentford",
    "https://www.whoscored.com/matches/1983631/live/england-premier-league-2026-2027-crystal-palace-nottingham-forest",
    "https://www.whoscored.com/matches/1983632/live/england-premier-league-2026-2027-wolverhampton-wanderers-brighton",
    "https://www.whoscored.com/matches/1983633/live/england-premier-league-2026-2027-leicester-burnley",
    "https://www.whoscored.com/matches/1983634/live/england-premier-league-2026-2027-bournemouth-sheffield-united",
    "https://www.whoscored.com/matches/1993935/live/spain-laliga-2026-2027-atletico-madrid-sevilla",
    "https://www.whoscored.com/matches/1993936/live/spain-laliga-2026-2027-real-sociedad-villarreal",
    "https://www.whoscored.com/matches/1993937/live/spain-laliga-2026-2027-athletic-club-osasuna",
    "https://www.whoscored.com/matches/1993938/live/spain-laliga-2026-2027-girona-mallorca",
    "https://www.whoscored.com/matches/1993939/live/spain-laliga-2026-2027-espanyol-las-palmas",
    "https://www.whoscored.com/matches/1993940/live/spain-laliga-2026-2027-real-madrid-atletico-madrid",
    "https://www.whoscored.com/matches/1993941/live/spain-laliga-2026-2027-sevilla-barcelona",
    "https://www.whoscored.com/matches/1993942/live/spain-laliga-2026-2027-villarreal-valencia",
    "https://www.whoscored.com/matches/1993943/live/spain-laliga-2026-2027-real-betis-real-sociedad",
    "https://www.whoscored.com/matches/1993944/live/spain-laliga-2026-2027-osasuna-celta-vigo",
    "https://www.whoscored.com/matches/1993945/live/spain-laliga-2026-2027-mallorca-getafe",
    "https://www.whoscored.com/matches/1993946/live/spain-laliga-2026-2027-rayo-vallecano-girona",
    "https://www.whoscored.com/matches/1993947/live/spain-laliga-2026-2027-deportivo-alaves-espanyol",
    "https://www.whoscored.com/matches/1993948/live/spain-laliga-2026-2027-las-palmas-levante",
    "https://www.whoscored.com/matches/1993949/live/spain-laliga-2026-2027-athletic-club-racing-santander",
    "https://www.whoscored.com/matches/1993950/live/spain-laliga-2026-2027-barcelona-atletico-madrid",
    "https://www.whoscored.com/matches/1993951/live/spain-laliga-2026-2027-real-madrid-sevilla",
    "https://www.whoscored.com/matches/1993952/live/spain-laliga-2026-2027-valencia-real-betis",
    "https://www.whoscored.com/matches/1993953/live/spain-laliga-2026-2027-real-sociedad-athletic-club",
    "https://www.whoscored.com/matches/1993954/live/spain-laliga-2026-2027-celta-vigo-villarreal",
    "https://www.whoscored.com/matches/1993955/live/spain-laliga-2026-2027-getafe-osasuna",
    "https://www.whoscored.com/matches/1993956/live/spain-laliga-2026-2027-girona-deportivo-alaves",
    "https://www.whoscored.com/matches/1993957/live/spain-laliga-2026-2027-espanyol-rayo-vallecano",
    "https://www.whoscored.com/matches/1993958/live/spain-laliga-2026-2027-levante-mallorca",
    "https://www.whoscored.com/matches/1993959/live/spain-laliga-2026-2027-racing-santander-las-palmas",
    "https://www.whoscored.com/matches/1993960/live/spain-laliga-2026-2027-atletico-madrid-valencia",
    "https://www.whoscored.com/matches/1993961/live/spain-laliga-2026-2027-sevilla-real-sociedad",
    "https://www.whoscored.com/matches/1993962/live/spain-laliga-2026-2027-villarreal-barcelona",
    "https://www.whoscored.com/matches/1993963/live/spain-laliga-2026-2027-real-betis-real-madrid",
    "https://www.whoscored.com/matches/1993964/live/spain-laliga-2026-2027-athletic-club-celta-vigo",
    "https://www.whoscored.com/matches/1993965/live/spain-laliga-2026-2027-osasuna-girona",
    "https://www.whoscored.com/matches/1993966/live/spain-laliga-2026-2027-mallorca-espanyol",
    "https://www.whoscored.com/matches/1993967/live/spain-laliga-2026-2027-rayo-vallecano-levante",
    "https://www.whoscored.com/matches/1993968/live/spain-laliga-2026-2027-deportivo-alaves-racing-santander",
    "https://www.whoscored.com/matches/1993969/live/spain-laliga-2026-2027-las-palmas-getafe",
    "https://www.whoscored.com/matches/1995412/live/germany-bundesliga-2026-2027-borussia-dortmund-bayer-leverkusen",
    "https://www.whoscored.com/matches/1995413/live/germany-bundesliga-2026-2027-vfb-stuttgart-eintracht-frankfurt",
    "https://www.whoscored.com/matches/1995414/live/germany-bundesliga-2026-2027-wolfsburg-borussia-m-gladbach",
    "https://www.whoscored.com/matches/1995415/live/germany-bundesliga-2026-2027-freiburg-hoffenheim",
    "https://www.whoscored.com/matches/1995416/live/germany-bundesliga-2026-2027-mainz-05-werder-bremen",
    "https://www.whoscored.com/matches/1995417/live/germany-bundesliga-2026-2027-fc-koln-st-pauli",
    "https://www.whoscored.com/matches/1995419/live/germany-bundesliga-2026-2027-paderborn-union-berlin",
    "https://www.whoscored.com/matches/1995420/live/germany-bundesliga-2026-2027-bayer-leverkusen-bayern-munich",
    "https://www.whoscored.com/matches/1995421/live/germany-bundesliga-2026-2027-rb-leipzig-vfb-stuttgart",
    "https://www.whoscored.com/matches/1995422/live/germany-bundesliga-2026-2027-eintracht-frankfurt-wolfsburg",
    "https://www.whoscored.com/matches/1995423/live/germany-bundesliga-2026-2027-borussia-m-gladbach-freiburg",
    "https://www.whoscored.com/matches/1995425/live/germany-bundesliga-2026-2027-hoffenheim-mainz-05",
    "https://www.whoscored.com/matches/1995427/live/germany-bundesliga-2026-2027-augsburg-fc-koln",
    "https://www.whoscored.com/matches/1995428/live/germany-bundesliga-2026-2027-st-pauli-elversberg",
    "https://www.whoscored.com/matches/1995429/live/germany-bundesliga-2026-2027-bayern-munich-eintracht-frankfurt",
    "https://www.whoscored.com/matches/1995430/live/germany-bundesliga-2026-2027-borussia-dortmund-rb-leipzig",
    "https://www.whoscored.com/matches/1995431/live/germany-bundesliga-2026-2027-vfb-stuttgart-bayer-leverkusen",
    "https://www.whoscored.com/matches/1995432/live/germany-bundesliga-2026-2027-wolfsburg-freiburg",
    "https://www.whoscored.com/matches/1995433/live/germany-bundesliga-2026-2027-borussia-m-gladbach-hoffenheim",
    "https://www.whoscored.com/matches/1995434/live/germany-bundesliga-2026-2027-mainz-05-union-berlin",
    "https://www.whoscored.com/matches/1995435/live/germany-bundesliga-2026-2027-werder-bremen-augsburg",
    "https://www.whoscored.com/matches/1995436/live/germany-bundesliga-2026-2027-fc-koln-paderborn",
    "https://www.whoscored.com/matches/1995437/live/germany-bundesliga-2026-2027-elversberg-bayer-leverkusen",
    "https://www.whoscored.com/matches/1995438/live/germany-bundesliga-2026-2027-st-pauli-bayern-munich",
    "https://www.whoscored.com/matches/1995439/live/germany-bundesliga-2026-2027-freiburg-borussia-dortmund",
    "https://www.whoscored.com/matches/1995440/live/germany-bundesliga-2026-2027-eintracht-frankfurt-rb-leipzig",
    "https://www.whoscored.com/matches/1995441/live/germany-bundesliga-2026-2027-hoffenheim-vfb-stuttgart",
    "https://www.whoscored.com/matches/1995442/live/germany-bundesliga-2026-2027-union-berlin-wolfsburg",
    "https://www.whoscored.com/matches/1995443/live/germany-bundesliga-2026-2027-augsburg-borussia-m-gladbach",
    "https://www.whoscored.com/matches/1995444/live/germany-bundesliga-2026-2027-paderborn-mainz-05",
    "https://www.whoscored.com/matches/1995445/live/germany-bundesliga-2026-2027-werder-bremen-st-pauli",
    "https://www.whoscored.com/matches/1984100/live/france-ligue-1-2026-2027-paris-saint-germain-marseille",
    "https://www.whoscored.com/matches/1984101/live/france-ligue-1-2026-2027-lyon-monaco",
    "https://www.whoscored.com/matches/1984102/live/france-ligue-1-2026-2027-lille-lens",
    "https://www.whoscored.com/matches/1984103/live/france-ligue-1-2026-2027-rennes-nice",
    "https://www.whoscored.com/matches/1984104/live/france-ligue-1-2026-2027-strasbourg-brest",
    "https://www.whoscored.com/matches/1984105/live/france-ligue-1-2026-2027-nantes-toulouse",
    "https://www.whoscored.com/matches/1984106/live/france-ligue-1-2026-2027-reims-montpellier",
    "https://www.whoscored.com/matches/1984107/live/france-ligue-1-2026-2027-saint-etienne-le-havre",
    "https://www.whoscored.com/matches/1984108/live/france-ligue-1-2026-2027-monaco-paris-saint-germain",
    "https://www.whoscored.com/matches/1984109/live/france-ligue-1-2026-2027-marseille-lyon",
    "https://www.whoscored.com/matches/1984110/live/france-ligue-1-2026-2027-lens-rennes",
    "https://www.whoscored.com/matches/1984111/live/france-ligue-1-2026-2027-nice-lille",
    "https://www.whoscored.com/matches/1984112/live/france-ligue-1-2026-2027-brest-nantes",
    "https://www.whoscored.com/matches/1984113/live/france-ligue-1-2026-2027-toulouse-strasbourg",
    "https://www.whoscored.com/matches/1984114/live/france-ligue-1-2026-2027-montpellier-auxerre",
    "https://www.whoscored.com/matches/1984115/live/france-ligue-1-2026-2027-le-mans-reims",
    "https://www.whoscored.com/matches/1984116/live/france-ligue-1-2026-2027-paris-fc-angers",
    "https://www.whoscored.com/matches/1984117/live/france-ligue-1-2026-2027-paris-saint-germain-lille",
    "https://www.whoscored.com/matches/1984118/live/france-ligue-1-2026-2027-lyon-lens",
    "https://www.whoscored.com/matches/1984119/live/france-ligue-1-2026-2027-monaco-nice",
    "https://www.whoscored.com/matches/1984120/live/france-ligue-1-2026-2027-marseille-rennes",
    "https://www.whoscored.com/matches/1984121/live/france-ligue-1-2026-2027-strasbourg-reims",
    "https://www.whoscored.com/matches/1984122/live/france-ligue-1-2026-2027-nantes-saint-etienne",
    "https://www.whoscored.com/matches/1984123/live/france-ligue-1-2026-2027-toulouse-brest",
    "https://www.whoscored.com/matches/1984124/live/france-ligue-1-2026-2027-le-havre-montpellier",
    "https://www.whoscored.com/matches/1984125/live/france-ligue-1-2026-2027-auxerre-paris-fc",
    "https://www.whoscored.com/matches/1984126/live/france-ligue-1-2026-2027-angers-le-mans",
    "https://www.whoscored.com/matches/1988101/live/italy-serie-a-2026-2027-inter-ac-milan",
    "https://www.whoscored.com/matches/1988102/live/italy-serie-a-2026-2027-juventus-napoli",
    "https://www.whoscored.com/matches/1988103/live/italy-serie-a-2026-2027-roma-lazio",
    "https://www.whoscored.com/matches/1988104/live/italy-serie-a-2026-2027-atalanta-fiorentina",
    "https://www.whoscored.com/matches/1988105/live/italy-serie-a-2026-2027-bologna-torino",
    "https://www.whoscored.com/matches/1988106/live/italy-serie-a-2026-2027-udinese-genoa",
    "https://www.whoscored.com/matches/1988107/live/italy-serie-a-2026-2027-COMO-parma",
    "https://www.whoscored.com/matches/1988108/live/italy-serie-a-2026-2027-cagliari-verona",
    "https://www.whoscored.com/matches/1988109/live/italy-serie-a-2026-2027-lecce-empoli",
    "https://www.whoscored.com/matches/1988110/live/italy-serie-a-2026-2027-sassuolo-palermo",
    "https://www.whoscored.com/matches/1988111/live/italy-serie-a-2026-2027-ac-milan-juventus",
    "https://www.whoscored.com/matches/1988112/live/italy-serie-a-2026-2027-napoli-inter",
    "https://www.whoscored.com/matches/1988113/live/italy-serie-a-2026-2027-lazio-atalanta",
    "https://www.whoscored.com/matches/1988114/live/italy-serie-a-2026-2027-fiorentina-roma",
    "https://www.whoscored.com/matches/1988115/live/italy-serie-a-2026-2027-torino-udinese",
    "https://www.whoscored.com/matches/1988116/live/italy-serie-a-2026-2027-genoa-bologna",
    "https://www.whoscored.com/matches/1988117/live/italy-serie-a-2026-2027-parma-cagliari",
    "https://www.whoscored.com/matches/1988118/live/italy-serie-a-2026-2027-verona-lecce",
    "https://www.whoscored.com/matches/1988119/live/italy-serie-a-2026-2027-empoli-sassuolo",
    "https://www.whoscored.com/matches/1988120/live/italy-serie-a-2026-2027-palermo-como",
    "https://www.whoscored.com/matches/1988121/live/italy-serie-a-2026-2027-inter-juventus",
    "https://www.whoscored.com/matches/1988122/live/italy-serie-a-2026-2027-ac-milan-napoli",
    "https://www.whoscored.com/matches/1988123/live/italy-serie-a-2026-2027-roma-atalanta",
    "https://www.whoscored.com/matches/1988124/live/italy-serie-a-2026-2027-lazio-fiorentina",
    "https://www.whoscored.com/matches/1988125/live/italy-serie-a-2026-2027-bologna-udinese",
    "https://www.whoscored.com/matches/1956063/live/brazil-brasileirão-2026-flamengo-palmeiras",
    "https://www.whoscored.com/matches/1956064/live/brazil-brasileirão-2026-botafogo-fluminense",
    "https://www.whoscored.com/matches/1956065/live/brazil-brasileirão-2026-atletico-mineiro-internacional",
    "https://www.whoscored.com/matches/1956066/live/brazil-brasileirão-2026-fortaleza-santos",
    "https://www.whoscored.com/matches/1956067/live/brazil-brasileirão-2026-red-bull-bragantino-juventude",
    "https://www.whoscored.com/matches/1956068/live/brazil-brasileirão-2026-ceara-sport-recife",
    "https://www.whoscored.com/matches/1956069/live/brazil-brasileirão-2026-palmeiras-botafogo",
    "https://www.whoscored.com/matches/1956070/live/brazil-brasileirão-2026-fluminense-flamengo",
    "https://www.whoscored.com/matches/1956071/live/brazil-brasileirão-2026-internacional-gremio",
    "https://www.whoscored.com/matches/1956072/live/brazil-brasileirão-2026-sao-paulo-corinthians",
    "https://www.whoscored.com/matches/1956073/live/brazil-brasileirão-2026-cruzeiro-atletico-mineiro",
    "https://www.whoscored.com/matches/1956074/live/brazil-brasileirão-2026-santos-vasco-da-gama",
    "https://www.whoscored.com/matches/1956075/live/brazil-brasileirão-2026-athletico-paranaense-fortaleza",
    "https://www.whoscored.com/matches/1956077/live/brazil-brasileirão-2026-bahia-vitoria",
    "https://www.whoscored.com/matches/1956078/live/brazil-brasileirão-2026-juventude-ceara",
    "https://www.whoscored.com/matches/1956079/live/brazil-brasileirão-2026-sport-recife-red-bull-bragantino",
    "https://www.whoscored.com/matches/1956080/live/brazil-brasileirão-2026-flamengo-sao-paulo",
    "https://www.whoscored.com/matches/1956081/live/brazil-brasileirão-2026-palmeiras-corinthians",
    "https://www.whoscored.com/matches/1956082/live/brazil-brasileirão-2026-botafogo-vasco-da-gama",
    "https://www.whoscored.com/matches/1956083/live/brazil-brasileirão-2026-fluminense-gremio",
    "https://www.whoscored.com/matches/1953077/live/usa-major-league-soccer-2026-los-angeles-fc-seattle-sounders-fc",
    "https://www.whoscored.com/matches/1953078/live/usa-major-league-soccer-2026-columbus-crew-atlanta-united",
    "https://www.whoscored.com/matches/1953079/live/usa-major-league-soccer-2026-philadelphia-union-dc-united",
    "https://www.whoscored.com/matches/1953080/live/usa-major-league-soccer-2026-portland-timbers-vancouver-whitecaps",
    "https://www.whoscored.com/matches/1953081/live/usa-major-league-soccer-2026-real-salt-lake-colorado-rapids",
    "https://www.whoscored.com/matches/1953082/live/usa-major-league-soccer-2026-houston-dynamo-st-louis-city",
    "https://www.whoscored.com/matches/1953083/live/usa-major-league-soccer-2026-minnesota-united-chicago-fire",
    "https://www.whoscored.com/matches/1953084/live/usa-major-league-soccer-2026-nashville-sc-charlotte-fc",
    "https://www.whoscored.com/matches/1953085/live/usa-major-league-soccer-2026-toronto-fc-cf-montreal",
    "https://www.whoscored.com/matches/1953086/live/usa-major-league-soccer-2026-san-jose-earthquakes-san-diego-fc",
    "https://www.whoscored.com/matches/1953087/live/usa-major-league-soccer-2026-inter-miami-cf-orlando-city",
    "https://www.whoscored.com/matches/1953088/live/usa-major-league-soccer-2026-la-galaxy-los-angeles-fc",
    "https://www.whoscored.com/matches/1953089/live/usa-major-league-soccer-2026-new-york-city-fc-red-bull-new-york",
    "https://www.whoscored.com/matches/1953090/live/usa-major-league-soccer-2026-fc-cincinnati-columbus-crew",
    "https://www.whoscored.com/matches/1953091/live/usa-major-league-soccer-2026-seattle-sounders-fc-portland-timbers",
    "https://www.whoscored.com/matches/1953092/live/usa-major-league-soccer-2026-austin-fc-houston-dynamo",
    "https://www.whoscored.com/matches/1953093/live/usa-major-league-soccer-2026-fc-dallas-colorado-rapids",
    "https://www.whoscored.com/matches/1953096/live/usa-major-league-soccer-2026-atlanta-united-nashville-sc",
    "https://www.whoscored.com/matches/1953097/live/usa-major-league-soccer-2026-charlotte-fc-dc-united",
    "https://www.whoscored.com/matches/1953098/live/usa-major-league-soccer-2026-chicago-fire-st-louis-city",
    "https://www.whoscored.com/matches/2029243/live/europe-champions-league-2026-2027-real-madrid-manchester-city",
    "https://www.whoscored.com/matches/2029244/live/europe-champions-league-2026-2027-barcelona-paris-saint-germain",
    "https://www.whoscored.com/matches/2029245/live/europe-champions-league-2026-2027-liverpool-inter",
    "https://www.whoscored.com/matches/2029246/live/europe-champions-league-2026-2027-arsenal-bayern-munich",
    "https://www.whoscored.com/matches/2029247/live/europe-champions-league-2026-2027-juventus-atletico-madrid",
    "https://www.whoscored.com/matches/2029248/live/europe-champions-league-2026-2027-borussia-dortmund-ac-milan",
    "https://www.whoscored.com/matches/2029249/live/europe-champions-league-2026-2027-bayer-leverkusen-napoli",
    "https://www.whoscored.com/matches/2029250/live/europe-champions-league-2026-2027-benfica-chelsea",
    "https://www.whoscored.com/matches/2029251/live/europe-champions-league-2026-2027-porto-rb-leipzig",
    "https://www.whoscored.com/matches/2029252/live/europe-champions-league-2026-2027-sporting-cp-aston-villa",
    "https://www.whoscored.com/matches/2029253/live/europe-champions-league-2026-2027-ajax-monaco",
    "https://www.whoscored.com/matches/2029254/live/europe-champions-league-2026-2027-psv-eindhoven-atalanta",
    "https://www.whoscored.com/matches/2029255/live/europe-champions-league-2026-2027-feyenoord-lille",
    "https://www.whoscored.com/matches/2029256/live/europe-champions-league-2026-2027-celtic-club-brugge",
    "https://www.whoscored.com/matches/2029257/live/europe-champions-league-2026-2027-galatasaray-vfb-stuttgart",
    "https://www.whoscored.com/matches/1998573/live/international-fifa-world-cup-2026-usa-mexico",
    "https://www.whoscored.com/matches/1998574/live/international-fifa-world-cup-2026-argentina-brazil",
    "https://www.whoscored.com/matches/1998575/live/international-fifa-world-cup-2026-england-germany",
    "https://www.whoscored.com/matches/1998576/live/international-fifa-world-cup-2026-portugal-netherlands",
    "https://www.whoscored.com/matches/1998577/live/international-fifa-world-cup-2026-italy-croatia",
    "https://www.whoscored.com/matches/1998578/live/international-fifa-world-cup-2026-belgium-morocco",
    "https://www.whoscored.com/matches/1998579/live/international-fifa-world-cup-2026-japan-south-korea",
    "https://www.whoscored.com/matches/1998580/live/international-fifa-world-cup-2026-senegal-nigeria",
    "https://www.whoscored.com/matches/1998581/live/international-fifa-world-cup-2026-uruguay-chile",
    "https://www.whoscored.com/matches/1998582/live/international-fifa-world-cup-2026-canada-australia",
    "https://www.whoscored.com/matches/1998583/live/international-fifa-world-cup-2026-denmark-poland",
    "https://www.whoscored.com/matches/1998584/live/international-fifa-world-cup-2026-serbia-austria",
    "https://www.whoscored.com/matches/1998585/live/international-fifa-world-cup-2026-ecuador-peru",
    "https://www.whoscored.com/matches/1998586/live/international-fifa-world-cup-2026-egypt-algeria",
    "https://www.whoscored.com/matches/1998587/live/international-fifa-world-cup-2026-ghana-cameroon",
    "https://www.whoscored.com/matches/1999628/live/turkey-super-lig-2026-2027-galatasaray-fenerbahce",
    "https://www.whoscored.com/matches/1999629/live/turkey-super-lig-2026-2027-besiktas-trabzonspor",
    "https://www.whoscored.com/matches/1999630/live/turkey-super-lig-2026-2027-istanbul-basaksehir-kasimpasa",
    "https://www.whoscored.com/matches/1999631/live/turkey-super-lig-2026-2027-samsunspor-antalyaspor",
    "https://www.whoscored.com/matches/1999632/live/turkey-super-lig-2026-2027-sivasspor-konyaspor",
    "https://www.whoscored.com/matches/1999633/live/turkey-super-lig-2026-2027-alanyaspor-rizespor",
    "https://www.whoscored.com/matches/1999634/live/turkey-super-lig-2026-2027-gaziantep-fk-kayserispor",
    "https://www.whoscored.com/matches/1999635/live/turkey-super-lig-2026-2027-goztepe-bodrum-fk",
    "https://www.whoscored.com/matches/1985906/live/belgium-jupiler-pro-league-2026-2027-club-brugge-anderlecht",
    "https://www.whoscored.com/matches/1985907/live/belgium-jupiler-pro-league-2026-2027-genk-royal-antwerp",
    "https://www.whoscored.com/matches/1985908/live/belgium-jupiler-pro-league-2026-2027-gent-union-st-gilloise",
    "https://www.whoscored.com/matches/1985909/live/belgium-jupiler-pro-league-2026-2027-standard-liege-charleroi",
    "https://www.whoscored.com/matches/1985910/live/belgium-jupiler-pro-league-2026-2027-cercle-brugge-westerlo",
    "https://www.whoscored.com/matches/1985911/live/belgium-jupiler-pro-league-2026-2027-sint-truiden-oh-leuven",
    "https://www.whoscored.com/matches/1985912/live/belgium-jupiler-pro-league-2026-2027-dender-beerschot",
    "https://www.whoscored.com/matches/1991104/live/russia-premier-league-2026-2027-zenit-st-petersburg-spartak-moscow",
    "https://www.whoscored.com/matches/1991105/live/russia-premier-league-2026-2027-cska-moscow-lokomotiv-moscow",
    "https://www.whoscored.com/matches/1991106/live/russia-premier-league-2026-2027-krasnodar-dynamo-moscow",
    "https://www.whoscored.com/matches/1991107/live/russia-premier-league-2026-2027-rostov-akhmat-grozny",
    "https://www.whoscored.com/matches/1991108/live/russia-premier-league-2026-2027-pari-nizhny-novgorod-khimki",
    "https://www.whoscored.com/matches/1991109/live/russia-premier-league-2026-2027-dynamo-makhachkala-akron-togliatti",
    "https://www.whoscored.com/matches/1995759/live/germany-2-bundesliga-2026-2027-hamburger-sv-schalke-04",
    "https://www.whoscored.com/matches/1995760/live/germany-2-bundesliga-2026-2027-hertha-berlin-hannover-96",
    "https://www.whoscored.com/matches/1995761/live/germany-2-bundesliga-2026-2027-fortuna-dusseldorf-kaiserslautern",
    "https://www.whoscored.com/matches/1995762/live/germany-2-bundesliga-2026-2027-karlsruher-sc-nurnberg",
    "https://www.whoscored.com/matches/1995763/live/germany-2-bundesliga-2026-2027-darmstadt-greuther-furth",
    "https://www.whoscored.com/matches/1995764/live/germany-2-bundesliga-2026-2027-heidenheim-bochum",
    "https://www.whoscored.com/matches/1995765/live/germany-2-bundesliga-2026-2027-holstein-kiel-magdeburg",
    "https://www.whoscored.com/matches/1969813/live/international-uefa-nations-league-b-2026-2027-kosovo-israel",
    "https://www.whoscored.com/matches/1969827/live/international-uefa-nations-league-a-2026-2027-serbia-germany",
    "https://www.whoscored.com/matches/1969790/live/international-uefa-nations-league-a-2026-2027-turkiye-belgium",
    "https://www.whoscored.com/matches/1993302/live/poland-ii-liga-2026-2027-podhale-nowy-targ-avia-swidnik"
]


def configure_logging(verbose: bool = False) -> None:
    """Configures structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def print_dataset_summary(dataset: ParsedMatchDataset, exported: ExportedFiles) -> None:
    """Prints a human-readable overview of the parsed match and exported files."""
    meta: dict[Any, Any] = (
        dataset.metadata_df.iloc[0].to_dict() if not dataset.metadata_df.empty else {}
    )
    home_team = meta.get("home_team_name", "Home")
    away_team = meta.get("away_team_name", "Away")
    score = meta.get("score", "N/A")

    print("\n" + "=" * 72)
    print(f"Match Summary: {home_team} vs {away_team} (Score: {score})")
    print("=" * 72)
    print(f"  Match ID         : {dataset.match_id}")
    for table_name, df in dataset.tables().items():
        if not df.empty:
            print(f"  {table_name:<16} : {len(df):,} rows ({len(df.columns)} cols)")

    if not dataset.events_df.empty and "event_type" in dataset.events_df.columns:
        print("\nTop 10 Event Types Captured:")
        counts = dataset.events_df["event_type"].value_counts(dropna=False).head(10)
        for ev_type, count in counts.items():
            print(f"  - {str(ev_type):<22}: {count:>5}")

    print("\nExported Files:")
    print(f"  [Raw JSON]       {exported.raw_json_path.resolve()}")
    for table_name, csv_path in exported.csv_files.items():
        label = f"[{table_name} CSV]"
        print(f"  {label:<16} {csv_path.resolve()}")
    if exported.parquet_files:
        parquet_dir = next(iter(exported.parquet_files.values())).parent.resolve()
        print(f"  [Parquet Dir]    {parquet_dir} ({len(exported.parquet_files)} tables)")
    print("=" * 72 + "\n")


def print_combined_summary(
    match_count: int,
    combined_dataset: ParsedMatchDataset,
    exported: ExportedFiles,
) -> None:
    """Prints a summary of the combined multi-match dataset and exported combined files."""
    print("\n" + "=" * 72)
    print(f"Combined Dataset Summary ({match_count} matches joined)")
    print("=" * 72)
    for table_name, df in combined_dataset.tables().items():
        if not df.empty:
            print(f"  {table_name:<16} : {len(df):,} rows ({len(df.columns)} cols)")

    print("\nExported Combined Files:")
    for table_name, csv_path in exported.csv_files.items():
        label = f"[{table_name} CSV]"
        print(f"  {label:<16} {csv_path.resolve()}")
    if exported.parquet_files:
        parquet_dir = next(iter(exported.parquet_files.values())).parent.resolve()
        print(f"  [Parquet Dir]    {parquet_dir} ({len(exported.parquet_files)} tables)")
    print("=" * 72 + "\n")


def process_match_payload(
    raw_data: dict[str, Any],
    parser: MatchDataParser,
    exporter: MatchDataExporter,
    raw_json_path: Path | None = None,
    overwrite_raw_json: bool = False,
) -> tuple[ParsedMatchDataset, ExportedFiles]:
    """Parses a raw `matchCentreData` dictionary and exports all configured formats."""
    dataset = parser.parse_all(raw_data)
    exported = exporter.export_all(
        raw_data,
        dataset,
        raw_json_path=raw_json_path,
        overwrite_raw_json=overwrite_raw_json,
    )
    print_dataset_summary(dataset, exported)
    return dataset, exported


def _normalize_url_list(urls: str | Sequence[str] | None) -> list[str]:
    """Normalizes a single URL string or sequence of URLs into a flat list of non-empty URLs."""
    if urls is None:
        return []
    if isinstance(urls, str):
        stripped = urls.strip()
        return [stripped] if stripped else []
    return [u.strip() for u in urls if u.strip()]


def run_pipeline(
    urls: str | Sequence[str] | None = None,
    from_json_files: Sequence[str | Path] | None = None,
    output_dir: str | Path = "output",
    processed_dir: str | Path = "data/processed",
    combined_dir: str | Path | None = None,
    export_combined: bool = True,
    headless: bool = False,
    force_scrape: bool = False,
) -> list[tuple[ParsedMatchDataset, ExportedFiles]]:
    """
    Runs the end-to-end scraping, parsing, and export pipeline for one or an array of URLs.
    Automatically reuses cached `data/raw/match_<id>_*.json` files unless `force_scrape=True`,
    and exports joined multi-match tables to `output/combined/` when `export_combined=True`.
    """
    parser = MatchDataParser()
    exporter = MatchDataExporter(
        output_dir=output_dir,
        processed_dir=processed_dir,
        combined_dir=combined_dir,
    )
    results: list[tuple[ParsedMatchDataset, ExportedFiles]] = []
    failed_items: list[tuple[str, str]] = []

    if from_json_files:
        for json_file in from_json_files:
            json_path = Path(json_file)
            try:
                raw_data = exporter.load_raw_json(json_path)
                results.append(
                    process_match_payload(raw_data, parser, exporter, raw_json_path=json_path)
                )
            except Exception as exc:
                logger.error("Skipping JSON file %s due to error: %s", json_path, exc)
                failed_items.append((str(json_path), str(exc)))

    normalized_urls = _normalize_url_list(urls)
    urls_to_scrape: list[str] = []
    for url in normalized_urls:
        match_id = WhoScoredScraper.extract_match_id_from_url(url)
        cached_json = None if force_scrape else exporter.find_cached_raw_json(match_id)
        if cached_json is not None:
            logger.info("Using cached raw JSON for matchId=%s: %s", match_id, cached_json)
            try:
                raw_data = exporter.load_raw_json(cached_json)
                results.append(
                    process_match_payload(raw_data, parser, exporter, raw_json_path=cached_json)
                )
            except Exception as exc:
                logger.error("Skipping cached JSON %s (%s) due to error: %s", cached_json, url, exc)
                failed_items.append((url, str(exc)))
        else:
            urls_to_scrape.append(url)

    if urls_to_scrape:
        total_to_scrape = len(urls_to_scrape)
        with WhoScoredScraper(headless=headless) as scraper:
            for idx, url in enumerate(urls_to_scrape, start=1):
                logger.info("[%d/%d] Scraping match URL: %s", idx, total_to_scrape, url)
                try:
                    raw_data = scraper.scrape_match(url)
                    results.append(
                        process_match_payload(
                            raw_data,
                            parser,
                            exporter,
                            overwrite_raw_json=force_scrape,
                        )
                    )
                except (WhoScoredScraperError, Exception) as exc:
                    logger.error(
                        "[%d/%d] Skipping URL (failed to scrape/process): %s -> %s",
                        idx,
                        total_to_scrape,
                        url,
                        exc,
                    )
                    failed_items.append((url, str(exc)))

    if export_combined and results:
        datasets = [ds for ds, _ in results]
        combined_ds, combined_exported = exporter.export_combined(datasets)
        print_combined_summary(len(datasets), combined_ds, combined_exported)

    if failed_items:
        print("\n" + "!" * 72)
        print(f"Skipped {len(failed_items)} Failed URL(s) / File(s):")
        print("!" * 72)
        for source, reason in failed_items:
            print(f"  - {source}\n    Reason: {reason}")
        print("!" * 72 + "\n")

    return results


def parse_cli_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(
        description="Scrape WhoScored match data and export Parquet + CSV tables for analytics."
    )
    cli.add_argument(
        "--url",
        "-u",
        nargs="+",
        help="One or more WhoScored Live match URLs (defaults to DEFAULT_MATCH_URLS).",
    )
    cli.add_argument(
        "--from-json",
        "-j",
        nargs="+",
        help="Parse one or more existing raw match JSON files offline instead of launching Chrome.",
    )
    cli.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory where per-match CSV files will be saved (default: 'output').",
    )
    cli.add_argument(
        "--combined-dir",
        "-c",
        default=None,
        help="Directory where combined CSV files will be saved (default: '<output-dir>/combined').",
    )
    cli.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip generating combined CSV/Parquet tables across all processed matches.",
    )
    cli.add_argument(
        "--processed-dir",
        "-p",
        default="data/processed",
        help="Directory where Parquet files will be saved (default: 'data/processed').",
    )
    cli.add_argument(
        "--force-scrape",
        action="store_true",
        help="Re-scrape URLs via Selenium even if a raw JSON backup already exists in data/raw/.",
    )
    cli.add_argument(
        "--headless",
        action="store_true",
        help="Run Chrome in headless mode (visible mode is recommended to avoid Cloudflare).",
    )
    cli.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return cli.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    configure_logging(verbose=args.verbose)

    target_urls = args.url
    if not target_urls and not args.from_json:
        target_urls = DEFAULT_MATCH_URLS

    run_pipeline(
        urls=target_urls,
        from_json_files=args.from_json,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
        combined_dir=args.combined_dir,
        export_combined=not args.no_combined,
        headless=args.headless,
        force_scrape=args.force_scrape,
    )
