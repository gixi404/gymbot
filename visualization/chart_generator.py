"""Statistical performance and weight chart generator for GymBot."""

import io
import logging
import re
import unicodedata
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Headless mode without graphical interface
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from storage.models import RawMessage

logger = logging.getLogger(__name__)

# Modern dark palette
BG_COLOR = "#0f172a"  # Slate 900
PANEL_COLOR = "#1e293b"  # Slate 800
GRID_COLOR = "#334155"  # Slate 700
TEXT_COLOR = "#f8fafc"  # Slate 50
SUBTEXT_COLOR = "#94a3b8"  # Slate 400
PRIMARY_ACCENT = "#38bdf8"  # Sky 400
SECONDARY_ACCENT = "#f59e0b"  # Amber 500
SUCCESS_ACCENT = "#10b981"  # Emerald 500
PURPLE_ACCENT = "#a855f7"  # Purple 500


class ChartGenerator:
    """Generates physical evolution and strength charts ready to send via Telegram."""

    @staticmethod
    def _apply_dark_style(fig, ax, title: str, xlabel: str = "Fecha", ylabel: str = "Kg"):
        """Applies a refined, readable dark theme to the Matplotlib figure."""
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
    def parse_weight_entries(cls, messages: list[RawMessage]) -> list[tuple[datetime, float]]:
        """Extracts all weight measurements with their respective date."""
        entries: list[tuple[datetime, float]] = []

        # Pattern for lines like: "07/10/26 - 70.05 kg" or "15/04/26 - 67.70 kg"
        multi_line_pattern = re.compile(
            r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-:]?\s*(\d{2,3}(?:[.,]\d{1,2})?)\s*kg",
            re.IGNORECASE,
        )

        for msg in messages:
            # 1. If the message contains records with the format 'DD/MM/YY - XX.XX kg'
            found_multiline = False
            for match in multi_line_pattern.finditer(msg.text):
                date_str, weight_str = match.groups()
                try:
                    # Normalize 2-digit year to 4 digits
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

            # 2. If it is not a multiline message but comes from the PesoCorporal group
            if not found_multiline and (msg.chat_title and "peso" in msg.chat_title.lower()):
                single_match = re.search(r"\b(\d{2,3}(?:[.,]\d{1,2})?)\s*(?:kg)?\b", msg.text)
                if single_match:
                    try:
                        weight = float(single_match.group(1).replace(",", "."))
                        if 40.0 <= weight <= 200.0:
                            entries.append((msg.date, weight))
                    except Exception:
                        pass

        # Sort chronologically and deduplicate by date (keeping the last one of the day)
        entries.sort(key=lambda x: x[0])
        deduped: dict[str, tuple[datetime, float]] = {}
        for dt, w in entries:
            key = dt.strftime("%Y-%m-%d")
            deduped[key] = (dt, w)

        return sorted(deduped.values(), key=lambda x: x[0])

    @classmethod
    def generate_weight_chart(cls, messages: list[RawMessage]) -> io.BytesIO | None:
        """Generates a body weight evolution chart with moving average and statistics."""
        points = cls.parse_weight_entries(messages)
        if len(points) < 2:
            return None

        dates = [p[0] for p in points]
        weights = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        try:
            cls._apply_dark_style(
                fig,
                ax,
                title="Evolución del Peso Corporal",
                xlabel="Fecha",
                ylabel="Peso (kg)",
            )

            # Measurement curve
            ax.plot(
                dates,
                weights,
                marker="o",
                markersize=4.5,
                color=PRIMARY_ACCENT,
                linewidth=1.8,
                label="Peso registrado",
                alpha=0.9,
            )

            # Moving average if there is enough data (>= 5 points)
            if len(weights) >= 5:
                window = min(4, len(weights) // 2)
                rolling_avg = []
                for i in range(len(weights)):
                    start_idx = max(0, i - window + 1)
                    subset = weights[start_idx : i + 1]
                    rolling_avg.append(sum(subset) / len(subset))

                ax.plot(
                    dates,
                    rolling_avg,
                    color=SECONDARY_ACCENT,
                    linewidth=2.2,
                    linestyle="-",
                    label="Tendencia suavizada",
                    alpha=0.95,
                )

            # Date format on the X axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            fig.autofmt_xdate(rotation=25)

            # Summary statistics in top box
            first_w, last_w = weights[0], weights[-1]
            min_w, max_w = min(weights), max(weights)
            diff = last_w - first_w
            diff_sign = "+" if diff > 0 else ""

            stats_box = (
                f"Inicio: {first_w:.1f} kg  |  Actual: {last_w:.1f} kg ({diff_sign}{diff:.1f} kg)\n"
                f"Mín: {min_w:.1f} kg  |  Máx: {max_w:.1f} kg"
            )
            ax.text(
                0.03,
                0.94,
                stats_box,
                transform=ax.transAxes,
                fontsize=8.5,
                color=TEXT_COLOR,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_COLOR, "edgecolor": GRID_COLOR, "alpha": 0.85},
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

    # Accessory exercises: if the header starts with one of these and the search
    # does not mention it, it is a false positive (e.g. "Gemelos en prensa" when
    # searching "prensa"). List curated against the real formats in the DB.
    _ACCESSORY_HEADS = ("gemelos", "gemelo", "antebrazo", "antebrazob")

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase without accents for robust matching ('Jalón' == 'jalon')."""
        decomposed = unicodedata.normalize("NFD", text.lower())
        return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")

    @staticmethod
    def _parse_set_block(text: str) -> tuple[float, list[int]] | None:
        """Extracts (weight_kg, [reps]) from a line like '145 kg 11.10.9', '60kg 10 8 8 8',
        '32.5 12 10 8 8' or '80kg 3x8'. Returns None if there is no valid block."""
        match = re.search(r"(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:kgs?|kh)?\s*(.*)$", text.strip(), re.IGNORECASE)
        if not match:
            return None
        try:
            weight = float(match.group(1).replace(",", "."))
        except ValueError:
            return None
        if not (2.0 <= weight <= 500.0):
            return None

        tail = match.group(2)
        cross = re.search(r"(\d{1,2})\s*[xX*]\s*(\d{1,2})", tail)
        if cross:
            reps = [int(cross.group(2))]
        else:
            reps = [int(n) for n in re.findall(r"\d{1,2}", tail)]
        reps = [r for r in reps if 1 <= r <= 60]
        if not reps:
            return None
        return weight, reps

    @classmethod
    def parse_exercise_data(
        cls, exercise_query: str, messages: list[RawMessage]
    ) -> tuple[str, list[tuple[datetime, float, float]]]:
        """Searches for an exercise and extracts (date, max_load, max_estimated_1rm) per session.

        Supports the real formats in the DB: lines that start with '-', '*' or
        directly the name followed by ':' or '('; sets on the same line or on
        the following lines.
        """
        query_norm = cls._normalize(exercise_query.strip())
        query_tokens = [t for t in re.split(r"\W+", query_norm) if t]
        matched_canonical = ""
        sessions: dict[str, tuple[datetime, float, float]] = {}

        for msg in messages:
            # Only workout chats
            if msg.chat_title and msg.chat_title.lower() in ("comidas", "pesocorporal"):
                continue

            lines = msg.text.split("\n")
            for idx, line in enumerate(lines):
                stripped = line.strip().lstrip("-*•").strip()
                if not stripped:
                    continue

                header_match = re.match(r"^([^:()\n]+?)\s*[:(]", stripped)
                if not header_match:
                    continue
                ex_name = header_match.group(1).strip()
                if not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", ex_name):
                    continue

                name_norm = cls._normalize(ex_name)
                if not all(token in name_norm for token in query_tokens):
                    continue

                # Exclude accessories (e.g. "Gemelos en prensa" when searching "prensa")
                first_word = re.split(r"\W+", name_norm)[0]
                if first_word in cls._ACCESSORY_HEADS and first_word not in query_tokens:
                    continue

                # Set block: first the same line, then the next 3 lines
                rest = stripped[header_match.end(1) :]
                block = cls._parse_set_block(rest)
                if block is None:
                    for nxt in lines[idx + 1 : idx + 4]:
                        block = cls._parse_set_block(nxt)
                        if block is not None:
                            break
                if block is None:
                    continue

                if not matched_canonical:
                    matched_canonical = ex_name

                weight, reps_list = block
                est_1rm = weight * (1.0 + max(reps_list) / 30.0)  # Epley
                session_key = msg.date.strftime("%Y-%m-%d")
                curr = sessions.get(session_key, (msg.date, 0.0, 0.0))
                sessions[session_key] = (
                    msg.date,
                    max(curr[1], weight),
                    max(curr[2], est_1rm),
                )

        result_list = sorted(sessions.values(), key=lambda x: x[0])
        return matched_canonical or exercise_query.title(), result_list

    @classmethod
    def generate_exercise_chart(cls, exercise_query: str, messages: list[RawMessage]) -> tuple[io.BytesIO, str] | None:
        """Generates a strength evolution chart (1RM and effective load) for an exercise."""
        canonical_name, data = cls.parse_exercise_data(exercise_query, messages)
        if len(data) < 2:
            return None

        dates = [d[0] for d in data]
        loads = [d[1] for d in data]
        e1rms = [d[2] for d in data]

        fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
        try:
            cls._apply_dark_style(
                fig,
                ax,
                title=f"Progresión de Fuerza: {canonical_name}",
                xlabel="Fecha",
                ylabel="Kg",
            )

            # Estimated 1RM line
            ax.plot(
                dates,
                e1rms,
                marker="o",
                markersize=4.5,
                color=SUCCESS_ACCENT,
                linewidth=2.0,
                label="1RM Estimado (Epley)",
                alpha=0.95,
            )

            # Effective Load line
            ax.plot(
                dates,
                loads,
                marker="s",
                markersize=4.0,
                color=PRIMARY_ACCENT,
                linewidth=1.6,
                linestyle="--",
                label="Carga de Trabajo (Kg)",
                alpha=0.85,
            )

            # Date format
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            fig.autofmt_xdate(rotation=25)

            # Statistics
            initial_1rm = e1rms[0]
            current_1rm = e1rms[-1]
            diff = current_1rm - initial_1rm
            sign = "+" if diff > 0 else ""

            stats_box = (
                f"1RM Inicial: {initial_1rm:.1f} kg  |  1RM Actual: {current_1rm:.1f} kg ({sign}{diff:.1f} kg)\n"
                f"Carga Inicial: {loads[0]:.1f} kg  |  Carga Actual: {loads[-1]:.1f} kg"
            )
            ax.text(
                0.03,
                0.94,
                stats_box,
                transform=ax.transAxes,
                fontsize=8.5,
                color=TEXT_COLOR,
                verticalalignment="top",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": BG_COLOR, "edgecolor": GRID_COLOR, "alpha": 0.85},
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
