#!/usr/bin/env python3
"""Interfaz de línea de comandos (CLI) interactiva y de paso de argumentos para GymBot."""

import argparse
import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from ai.analyzer import GymAnalyzer
from config.settings import get_settings
from storage.database import GymDatabase
from telegram_service.mtproto_reader import TelegramWorkoutReader

console = Console()


def display_welcome_banner(database: GymDatabase) -> None:
    """Muestra el banner inicial con estadísticas locales en la CLI."""
    stats = database.get_stats()
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(style="bold yellow")
    stats_table.add_column(style="bold white")

    stats_table.add_row("Mensajes en BD:", str(stats.total_messages))
    stats_table.add_row(
        "Rango de Fechas:",
        f"{stats.first_message_date.strftime('%d/%m/%Y') if stats.first_message_date else 'N/A'} - "
        f"{stats.last_message_date.strftime('%d/%m/%Y') if stats.last_message_date else 'N/A'}",
    )
    stats_table.add_row(
        "Última Sincronización:",
        stats.last_sync_at.strftime("%d/%m/%Y %H:%M") if stats.last_sync_at else "Pendiente",
    )

    panel = Panel(
        stats_table,
        title="[bold green]🏋️ GYMBOT - Analizador Inteligente de Entrenamientos[/bold green]",
        subtitle="[dim]Comandos: :sync, :stats, :clear, :help, :exit[/dim]",
        border_style="cyan",
    )
    console.print(panel)


def display_help() -> None:
    """Imprime el listado de comandos y ejemplos de uso en la CLI."""
    table = Table(title="Comandos Disponibles en la CLI", border_style="cyan")
    table.add_column("Comando", style="bold yellow")
    table.add_column("Descripción", style="white")

    table.add_row(":sync", "Sincroniza los últimos entrenamientos desde el chat de Telegram")
    table.add_row(":stats", "Muestra estadísticas del historial local de entrenamientos")
    table.add_row(":clear", "Limpia la pantalla de la terminal")
    table.add_row(":help", "Muestra este mensaje de ayuda")
    table.add_row(":exit / :quit", "Sale de la aplicación interactiva")
    table.add_row("<cualquier texto>", "Envía la pregunta a Gemini 2.5 Flash para su análisis")

    console.print(table)


async def perform_sync(reader: TelegramWorkoutReader) -> None:
    """Ejecuta la sincronización con MTProto mostrando un spinner visual."""
    with console.status("[bold cyan]Sincronizando entrenamientos desde Telegram MTProto...", spinner="dots"):
        try:
            new_msgs, total_msgs = await reader.sync_channel()
            console.print(
                f"[bold green]✔ Sincronización exitosa:[/bold green] "
                f"[cyan]{new_msgs}[/cyan] nuevos mensajes guardados. "
                f"Total en base de datos: [cyan]{total_msgs}[/cyan]."
            )
        except Exception as e:
            console.print(f"[bold red]✖ Error en la sincronización:[/bold red] {e}")


def display_stats(database: GymDatabase) -> None:
    """Muestra una tabla detallada con estadísticas de la base de datos."""
    stats = database.get_stats()
    table = Table(title="📊 Estadísticas Locales de Entrenamientos", border_style="green")
    table.add_column("Métrica", style="bold yellow")
    table.add_column("Valor", style="white")

    table.add_row("Total de Registros Guardados", str(stats.total_messages))
    table.add_row(
        "Fecha del Primer Entrenamiento",
        stats.first_message_date.strftime("%Y-%m-%d %H:%M") if stats.first_message_date else "N/A",
    )
    table.add_row(
        "Fecha del Último Entrenamiento",
        stats.last_message_date.strftime("%Y-%m-%d %H:%M") if stats.last_message_date else "N/A",
    )
    table.add_row(
        "Última Sincronización Realizada",
        stats.last_sync_at.strftime("%Y-%m-%d %H:%M") if stats.last_sync_at else "Nunca",
    )
    console.print(table)


def process_query(
    analyzer: GymAnalyzer,
    query: str,
    reader: Optional[TelegramWorkoutReader] = None,
    database: Optional[GymDatabase] = None,
) -> None:
    """Envía la pregunta al analizador de IA y presenta la respuesta formateada."""
    with console.status("Pensando...", spinner="dots"):
        if reader:
            try:
                asyncio.run(reader.sync_channel())
            except Exception:
                pass

        try:
            active_memory = database.get_active_memory("cli_user", limit=8) if database else None
            response = analyzer.analyze_query(query, conversation_history=active_memory)
            if database:
                database.add_conversation_message("cli_user", "user", query)
                database.add_conversation_message("cli_user", "model", response)
        except Exception as e:
            console.print(f"[bold red]Error analizando consulta:[/bold red] {e}")
            return

    console.print()
    console.print(Markdown(response))
    console.print()


async def interactive_mode(database: GymDatabase, reader: TelegramWorkoutReader, analyzer: GymAnalyzer) -> None:
    """Bucle principal de la terminal interactiva."""
    display_welcome_banner(database)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]gymbot[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Saliendo de GymBot... ¡A seguir entrenando![/yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in (":exit", ":quit", "exit", "quit"):
            console.print("[yellow]Saliendo de GymBot... ¡A seguir entrenando![/yellow]")
            break
        elif cmd in (":sync", "sync"):
            await perform_sync(reader)
        elif cmd in (":stats", "stats"):
            display_stats(database)
        elif cmd in (":clear", "clear"):
            database.reset_active_memory("cli_user")
            console.clear()
            display_welcome_banner(database)
            console.print("[dim green]✔ Memoria de conversación reiniciada (historial preservado).[/dim green]\n")
        elif cmd in (":help", "help"):
            display_help()
        else:
            process_query(analyzer, user_input, reader, database)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GymBot CLI: Analizador inteligente de entrenamientos de gimnasio desde Telegram."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Consulta directa para el analizador o comando especial (:sync, :stats)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Fuerza la sincronización con Telegram y finaliza",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Muestra las estadísticas de la base de datos y finaliza",
    )

    args = parser.parse_args()

    try:
        settings = get_settings()
        database = GymDatabase(settings.DATABASE_PATH)
        reader = TelegramWorkoutReader(settings, database)
        analyzer = GymAnalyzer(settings, database)
    except Exception as e:
        console.print(f"[bold red]Error de inicialización:[/bold red] {e}")
        console.print("[yellow]Verifica que el archivo .env esté configurado correctamente.[/yellow]")
        sys.exit(1)

    # Modo paso de comandos directos por flags
    if args.sync:
        asyncio.run(perform_sync(reader))
        return

    if args.stats:
        display_stats(database)
        return

    # Modo argumento directo (ej. python cli.py "¿Cuál fue mi mejor press de banca?")
    if args.query:
        query_text = " ".join(args.query).strip()
        if query_text.lower() in (":sync", "sync"):
            asyncio.run(perform_sync(reader))
        elif query_text.lower() in (":stats", "stats"):
            display_stats(database)
        elif query_text.lower() in (":help", "help"):
            display_help()
        else:
            process_query(analyzer, query_text, reader, database)
        return

    # Modo interactivo si no se pasaron argumentos
    asyncio.run(interactive_mode(database, reader, analyzer))


if __name__ == "__main__":
    main()
