"""Generador de gráficos estadísticos de rendimiento y peso para GymBot."""

import io
import logging
import re
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Modo headless sin interfaz gráfica
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from storage.models import RawMessage

logger = logging.getLogger(__name__)

# Paleta oscura moderna
BG_COLOR = "#0f172a"        # Slate 900
PANEL_COLOR = "#1e293b"     # Slate 800
GRID_COLOR = "#334155"      # Slate 700
TEXT_COLOR = "#f8fafc"      # Slate 50
SUBTEXT_COLOR = "#94a3b8"   # Slate 400
PRIMARY_ACCENT = "#38bdf8"  # Sky 400
SECONDARY_ACCENT = "#f59e0b"# Amber 500
SUCCESS_ACCENT = "#10b981"  # Emerald 500
PURPLE_ACCENT = "#a855f7"   # Purple 500


class ChartGenerator:
    """Genera gráficos de evolución física y fuerza listos para enviar por Telegram."""

    @staticmethod
    def _apply_dark_style(fig, ax, title: str, xlabel: str = "Fecha", ylabel: str = "Kg"):
        """Aplica un tema oscuro refinado y legible a la figura de Matplotlib."""
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(PANEL_COLOR)

        ax.set_title(title, color=TEXT_COLOR, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, color=SUBTEXT_COLOR, fontsize=10, labelpad=8)
        ax.set_ylabel(ylabel, color=SUBTEXT_COLOR, fontsize=10, labelpad=8)

        ax.tick_params(colors=SUBTEXT_COLOR, labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.35, color=GRID_COLOR)

        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
            spine.set_alpha(0.5)

    @classmethod
    def parse_weight_entries(cls, messages: List[RawMessage]) -> List[Tuple[datetime, float]]:
        """Extrae todas las mediciones de peso con su fecha respectiva."""
        entries: List[Tuple[datetime, float]] = []

        # Patrón para líneas tipo: "07/10/26 - 70.05 kg" o "15/04/26 - 67.70 kg"
        multi_line_pattern = re.compile(
            r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-:]?\s*(\d{2,3}(?:[.,]\d{1,2})?)\s*kg",
            re.IGNORECASE,
        )

        for msg in messages:
            # 1. Si el mensaje contiene registros con formato 'DD/MM/AA - XX.XX kg'
            found_multiline = False
            for match in multi_line_pattern.finditer(msg.text):
                date_str, weight_str = match.groups()
                try:
                    # Normalizar año de 2 dígitos a 4 dígitos
                    parts = date_str.split("/")
                    day, month = int(parts[0]), int(parts[1])
                    year = int(parts[2])
                    if year < 100:
                        year += 2000
                    dt = datetime(year, month, day)
                    weight = float(weight_str.replace(",", "."))
                    if 40.0 <= weight <= 200.0:
                        entries.append((dt, weight))
                        found_multiline = True
                except Exception:
                    continue

            # 2. Si no es un mensaje multilínea pero proviene del grupo PesoCorporal
            if not found_multiline and (msg.chat_title and "peso" in msg.chat_title.lower()):
                single_match = re.search(r"\b(\d{2,3}(?:[.,]\d{1,2})?)\s*(?:kg)?\b", msg.text)
                if single_match:
                    try:
                        weight = float(single_match.group(1).replace(",", "."))
                        if 40.0 <= weight <= 200.0:
                            entries.append((msg.date, weight))
                    except Exception:
                        pass

        # Ordenar cronológicamente y deduplicar por fecha (manteniendo el último del día)
        entries.sort(key=lambda x: x[0])
        deduped: dict[str, Tuple[datetime, float]] = {}
        for dt, w in entries:
            key = dt.strftime("%Y-%m-%d")
            deduped[key] = (dt, w)

        return sorted(list(deduped.values()), key=lambda x: x[0])

    @classmethod
    def generate_weight_chart(cls, messages: List[RawMessage]) -> Optional[io.BytesIO]:
        """Genera un gráfico de evolución de peso corporal con media móvil y estadísticas."""
        points = cls.parse_weight_entries(messages)
        if len(points) < 2:
            return None

        dates = [p[0] for p in points]
        weights = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        try:
            cls._apply_dark_style(
                fig, ax,
                title="Evolución del Peso Corporal",
                xlabel="Fecha",
                ylabel="Peso (kg)",
            )

            # Curva de mediciones
            ax.plot(
                dates, weights,
                marker="o", markersize=4.5,
                color=PRIMARY_ACCENT, linewidth=1.8,
                label="Peso registrado",
                alpha=0.9,
            )

            # Media móvil si hay suficientes datos (>= 5 puntos)
            if len(weights) >= 5:
                window = min(4, len(weights) // 2)
                rolling_avg = []
                for i in range(len(weights)):
                    start_idx = max(0, i - window + 1)
                    subset = weights[start_idx : i + 1]
                    rolling_avg.append(sum(subset) / len(subset))

                ax.plot(
                    dates, rolling_avg,
                    color=SECONDARY_ACCENT, linewidth=2.2, linestyle="-",
                    label=f"Tendencia suavizada",
                    alpha=0.95,
                )

            # Formato de fechas en eje X
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            fig.autofmt_xdate(rotation=25)

            # Estadísticas resumen en caja superior
            first_w, last_w = weights[0], weights[-1]
            min_w, max_w = min(weights), max(weights)
            diff = last_w - first_w
            diff_sign = "+" if diff > 0 else ""

            stats_box = (
                f"Inicio: {first_w:.1f} kg  |  Actual: {last_w:.1f} kg ({diff_sign}{diff:.1f} kg)\n"
                f"Mín: {min_w:.1f} kg  |  Máx: {max_w:.1f} kg"
            )
            ax.text(
                0.03, 0.94, stats_box,
                transform=ax.transAxes,
                fontsize=8.5,
                color=TEXT_COLOR,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_COLOR, edgecolor=GRID_COLOR, alpha=0.85),
            )

            ax.legend(
                loc="lower right",
                facecolor=BG_COLOR,
                edgecolor=GRID_COLOR,
                fontsize=8.5,
                labelcolor=TEXT_COLOR,
            )

            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
            buf.seek(0)
            return buf
        finally:
            plt.close(fig)

    @classmethod
    def parse_exercise_data(
        cls, exercise_query: str, messages: List[RawMessage]
    ) -> Tuple[str, List[Tuple[datetime, float, float]]]:
        """Busca un ejercicio y extrae (fecha, carga_maxima, 1rm_estimado_maximo) por sesión."""
        query_norm = exercise_query.strip().lower()
        matched_canonical = ""
        sessions: dict[str, Tuple[datetime, float, float]] = {}

        for msg in messages:
            # Solo chats de entrenamientos
            if msg.chat_title and msg.chat_title.lower() in ("comidas", "pesocorporal"):
                continue

            lines = msg.text.split("\n")
            for idx, line in enumerate(lines):
                line_clean = line.strip()
                if not line_clean.startswith("-"):
                    continue

                ex_header = line_clean.lstrip("-").split(":")[0].strip()
                ex_name_clean = re.sub(r"\(.*?\)", "", ex_header).strip()

                if query_norm in ex_name_clean.lower() or any(w in ex_name_clean.lower() for w in query_norm.split()):
                    if not matched_canonical:
                        matched_canonical = ex_name_clean

                    # Buscar las líneas siguientes con las series
                    sub_text = " ".join(lines[idx + 1 : idx + 4])

                    # Caso A: '53 kg 13.12.11' o '80 kg 12.10.8' (formato común del usuario)
                    dot_match = re.search(
                        r"(\d+(?:[.,]\d+)?)\s*kg\s+(\d+(?:[.,]\d+)+)",
                        sub_text,
                        re.IGNORECASE,
                    )
                    if dot_match:
                        weight = float(dot_match.group(1).replace(",", "."))
                        reps_str = dot_match.group(2)
                        reps_list = [int(r) for r in re.split(r"[.,]", reps_str) if r.isdigit()]
                        if reps_list:
                            max_rep = max(reps_list)
                            # Epley: w * (1 + r/30)
                            est_1rm = weight * (1.0 + max_rep / 30.0)
                            session_key = msg.date.strftime("%Y-%m-%d")
                            curr = sessions.get(session_key, (msg.date, 0.0, 0.0))
                            sessions[session_key] = (
                                msg.date,
                                max(curr[1], weight),
                                max(curr[2], est_1rm),
                            )
                        continue

                    # Caso B: '80kg 3x8' o '80 x 8'
                    std_match = re.search(
                        r"(\d+(?:[.,]\d+)?)\s*(?:kg)?\s*[xX*]\s*(\d+)",
                        sub_text,
                    )
                    if std_match:
                        weight = float(std_match.group(1).replace(",", "."))
                        reps = int(std_match.group(2))
                        est_1rm = weight * (1.0 + reps / 30.0)
                        session_key = msg.date.strftime("%Y-%m-%d")
                        curr = sessions.get(session_key, (msg.date, 0.0, 0.0))
                        sessions[session_key] = (
                            msg.date,
                            max(curr[1], weight),
                            max(curr[2], est_1rm),
                        )

        result_list = sorted(list(sessions.values()), key=lambda x: x[0])
        return matched_canonical or exercise_query.title(), result_list

    @classmethod
    def generate_exercise_chart(
        cls, exercise_query: str, messages: List[RawMessage]
    ) -> Optional[Tuple[io.BytesIO, str]]:
        """Genera un gráfico de evolución de fuerza (1RM y carga efectiva) para un ejercicio."""
        canonical_name, data = cls.parse_exercise_data(exercise_query, messages)
        if len(data) < 2:
            return None

        dates = [d[0] for d in data]
        loads = [d[1] for d in data]
        e1rms = [d[2] for d in data]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        try:
            cls._apply_dark_style(
                fig, ax,
                title=f"Progresión de Fuerza: {canonical_name}",
                xlabel="Fecha",
                ylabel="Kg",
            )

            # Línea de 1RM Estimado
            ax.plot(
                dates, e1rms,
                marker="o", markersize=4.5,
                color=SUCCESS_ACCENT, linewidth=2.0,
                label="1RM Estimado (Epley)",
                alpha=0.95,
            )

            # Línea de Carga Efectiva
            ax.plot(
                dates, loads,
                marker="s", markersize=4.0,
                color=PRIMARY_ACCENT, linewidth=1.6, linestyle="--",
                label="Carga de Trabajo (Kg)",
                alpha=0.85,
            )

            # Formato de fechas
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            fig.autofmt_xdate(rotation=25)

            # Estadísticas
            initial_1rm = e1rms[0]
            current_1rm = e1rms[-1]
            diff = current_1rm - initial_1rm
            sign = "+" if diff > 0 else ""

            stats_box = (
                f"1RM Inicial: {initial_1rm:.1f} kg  |  1RM Actual: {current_1rm:.1f} kg ({sign}{diff:.1f} kg)\n"
                f"Carga Inicial: {loads[0]:.1f} kg  |  Carga Actual: {loads[-1]:.1f} kg"
            )
            ax.text(
                0.03, 0.94, stats_box,
                transform=ax.transAxes,
                fontsize=8.5,
                color=TEXT_COLOR,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_COLOR, edgecolor=GRID_COLOR, alpha=0.85),
            )

            ax.legend(
                loc="lower right",
                facecolor=BG_COLOR,
                edgecolor=GRID_COLOR,
                fontsize=8.5,
                labelcolor=TEXT_COLOR,
            )

            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
            buf.seek(0)
            return buf, canonical_name
        finally:
            plt.close(fig)
