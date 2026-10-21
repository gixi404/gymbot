#!/usr/bin/env python3
"""Live diagnostic for GymBot: checks query classification and answer generation.

Runs representative queries through the real classifier and LLM engines using
the local database and the credentials in .env. Read-only: it never writes to
the database (no conversation memory is persisted).

Usage:
    python diagnose_bot.py                 # classification only (cheap, fast)
    python diagnose_bot.py --respuesta     # also generate the full answer
    python diagnose_bot.py -q "cuanta proteina comi hoy?" --respuesta
"""

import argparse
import time

from ai.analyzer import GymAnalyzer
from ai.router import classify_query, resolve_group_targets
from config.settings import get_settings
from storage.database import GymDatabase

# (query, expected groups). Empty set means "any route" (no assertion).
CASES: list[tuple[str, set[str]]] = [
    ("¿Cuál fue mi mejor press de banca este mes?", {"Sesiones"}),
    ("¿Cuánta proteína comí hoy?", {"Comidas"}),
    ("¿Cómo viene mi peso corporal en las últimas semanas?", {"PesoCorporal"}),
    ("¿Cuánto cardio hice esta semana?", {"Cardio"}),
    ("Comparame el progreso de sentadilla de los últimos 3 meses y mi peso", {"Sesiones", "PesoCorporal"}),
    ("¿Qué comí ayer?", {"Comidas"}),
    ("Haceme un resumen completo de todo mi historial", set()),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico en vivo del clasificador y las respuestas de GymBot.")
    parser.add_argument("--query", "-q", help="Probar una sola consulta en vez de la lista de casos.")
    parser.add_argument("--respuesta", "-r", action="store_true", help="Generar también la respuesta completa.")
    args = parser.parse_args()

    settings = get_settings()
    database = GymDatabase(settings.DATABASE_PATH)
    analyzer = GymAnalyzer(settings, database)

    stats = database.get_stats()
    print(f"DB: {stats.total_messages} mensajes  ({settings.DATABASE_PATH})\n")

    cases = [(args.query, set())] if args.query else CASES

    ok = 0
    for query, expected in cases:
        t0 = time.perf_counter()
        route = classify_query(query, analyzer.llm_client)
        titles, _ = resolve_group_targets(route.groups, settings)
        dt = (time.perf_counter() - t0) * 1000

        got = set(route.groups)
        verdict = "OK" if (not expected or expected <= got) else "MISMATCH"
        ok += verdict == "OK"
        print(f"[{verdict}] {query}")
        print(
            f"    complexity={route.complexity}  groups={route.groups}  days_lookback={route.days_lookback}  ({dt:.0f} ms)"
        )
        if expected:
            print(f"    esperado~ {sorted(expected)}  |  filtros SQL: {titles}")

        if args.respuesta:
            t1 = time.perf_counter()
            answer = analyzer.analyze_query(query, route=route)
            dt1 = (time.perf_counter() - t1) * 1000
            flat = " ".join(answer.split())
            print(f"    respuesta ({dt1:.0f} ms, {len(answer)} chars): {flat[:300]}{'...' if len(flat) > 300 else ''}")
        print()

    print(f"Clasificación: {ok}/{len(cases)} en el grupo esperado.")


if __name__ == "__main__":
    main()
