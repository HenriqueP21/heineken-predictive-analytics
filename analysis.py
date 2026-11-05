from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

# Configura o backend do matplotlib para não exigir interface gráfica.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from mysql.connector import Error as MySQLError  # noqa: E402

from main import Database, Settings, load_settings  # noqa: E402

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CAMINHOS, CONSTANTES E CONFIGURAÇÕES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = BASE_DIR / "reports" / "figures"

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 30

# Colunas usadas na análise exploratória (features + preço + target).
ANALYTICAL_COLUMNS: tuple[str, ...] = (
    "close_price",
    "volume",
    "return_1d",
    "return_5d",
    "return_20d",
    "sma_5",
    "sma_10",
    "sma_20",
    "volatility_5d",
    "volatility_20d",
    "price_sma5_ratio",
    "price_sma20_ratio",
    "target_next_close",
)

# Colunas usadas para calcular a matriz de correlação (features + preço + target).
CORRELATION_COLUMNS: tuple[str, ...] = ANALYTICAL_COLUMNS

# Campos de estatísticas descritivas, com rótulo e formato de exibição.
STAT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("start_date", "Início da série", "{}"),
    ("end_date", "Fim da série", "{}"),
    ("observations", "Observações de retorno", "{:,.0f}"),
    ("cumulative_return", "Retorno acumulado no período", "{:+.2%}"),
    ("mean_return_1d", "Retorno diário médio", "{:+.4%}"),
    ("median_return_1d", "Retorno diário mediano", "{:+.4%}"),
    ("std_return_1d", "Desvio-padrão do retorno diário", "{:.4%}"),
    ("annualized_return", "Retorno anualizado (média x 252)", "{:+.2%}"),
    ("annualized_volatility", "Volatilidade anualizada", "{:.2%}"),
    ("min_return_1d", "Pior retorno diário", "{:+.2%}"),
    ("max_return_1d", "Melhor retorno diário", "{:+.2%}"),
    ("var_95_1d", "VaR histórico 95% (1 dia)", "{:+.2%}"),
    ("skewness", "Assimetria dos retornos", "{:.3f}"),
    ("excess_kurtosis", "Curtose em excesso", "{:.3f}"),
    ("max_drawdown", "Drawdown máximo", "{:.2%}"),
    ("mean_volatility_20d", "Volatilidade 20d média", "{:.4%}"),
    ("mean_volume", "Volume médio", "{:,.0f}"),
    ("median_volume", "Volume mediano", "{:,.0f}"),
    ("max_volume", "Volume máximo", "{:,.0f}"),
    ("zero_volume_days", "Dias com volume zero", "{:,.0f}"),
)


# ============================================================
# LOGGING (LOGGER)
# ============================================================

def _build_logger() -> logging.Logger:
    """
    Cria o logger do módulo com arquivo próprio (logs/analysis.log)
    e saída no console, no mesmo formato do pipeline principal.

    `propagate=False` evita linhas duplicadas, já que o main.py
    configura o logger raiz ao ser importado.
    """

    log = logging.getLogger("heineken_analysis")

    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    log.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "analysis.log",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    return log


logger = _build_logger()


# ============================================================
# EXCEÇÕES DE ANÁLISE
# ============================================================

class AnalysisError(RuntimeError):
    """Erro de negócio/dados na etapa de análise exploratória."""


# ============================================================
# FUNÇÕES DE LEITURA DE DADOS
# ============================================================

def _fetch_dataframe(
    db: Database,
    query: str,
    params: tuple | list,
) -> pd.DataFrame:
    """Executa uma consulta de leitura e devolve um DataFrame."""

    cursor = db.cursor()

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
    except MySQLError as exc:
        raise AnalysisError(
            f"Falha ao executar consulta no MySQL: {exc}"
        ) from exc
    finally:
        cursor.close()

    return pd.DataFrame(rows)


def load_analytical_frame(
    db: Database,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """
    Lê `analytical_data` (com ticker e moeda vindos de `assets`).

    Retorna um DataFrame ordenado por (ticker, price_date), com
    colunas numéricas convertidas para float.
    """

    placeholders = ", ".join(["%s"] * len(tickers))
    select_columns = ",\n            ".join(
        f"ad.{column}" for column in ANALYTICAL_COLUMNS
    )

    # Consulta SQL para ler os dados analíticos, juntando com a tabela de ativos.
    query = f"""
        SELECT
            a.ticker,
            a.currency,
            ad.price_date,
            {select_columns}
        FROM analytical_data AS ad
        INNER JOIN assets AS a
            ON a.id = ad.asset_id
        WHERE a.ticker IN ({placeholders})
          AND a.active = TRUE
        ORDER BY a.ticker, ad.price_date
    """

    df = _fetch_dataframe(db, query, tickers)

    if df.empty:
        raise AnalysisError(
            "A tabela analytical_data não retornou registros para: "
            + ", ".join(tickers)
            + ". Execute antes o ETL (python main.py --mode etl)."
        )

    df["price_date"] = pd.to_datetime(df["price_date"])

    for column in ANALYTICAL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values(["ticker", "price_date"]).reset_index(drop=True)

    logger.info(
        "analytical_data carregada: %s linhas, %s ativos.",
        f"{len(df):,}",
        df["ticker"].nunique(),
    )

    return df


def load_adjusted_close(
    db: Database,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """Lê o fechamento ajustado de `market_prices`."""

    placeholders = ", ".join(["%s"] * len(tickers))

    query = f"""
        SELECT
            a.ticker,
            mp.price_date,
            mp.adjusted_close
        FROM market_prices AS mp
        INNER JOIN assets AS a
            ON a.id = mp.asset_id
        WHERE a.ticker IN ({placeholders})
          AND a.active = TRUE
        ORDER BY a.ticker, mp.price_date
    """

    df = _fetch_dataframe(db, query, tickers)

    if df.empty:
        return pd.DataFrame(
            columns=["ticker", "price_date", "adjusted_close"]
        )

    df["price_date"] = pd.to_datetime(df["price_date"])
    df["adjusted_close"] = pd.to_numeric(
        df["adjusted_close"],
        errors="coerce",
    )

    return df


def attach_adjusted_close(
    db: Database,
    df: pd.DataFrame,
    tickers: tuple[str, ...],
) -> pd.DataFrame:
    """
    Adiciona a coluna `adjusted_close` ao DataFrame analítico.

    Se um ativo não tiver nenhum valor ajustado (ou a leitura de
    `market_prices` falhar), usa `close_price` como alternativa
    e registra um aviso.
    """

    try:
        adjusted = load_adjusted_close(db, tickers)
    except AnalysisError as exc:
        logger.warning(
            "Não foi possível ler adjusted_close: %s. "
            "Usando close_price.",
            exc,
        )
        adjusted = pd.DataFrame(
            columns=["ticker", "price_date", "adjusted_close"]
        )

    if adjusted.empty:
        df = df.assign(adjusted_close=np.nan)
    else:
        df = df.merge(
            adjusted,
            on=["ticker", "price_date"],
            how="left",
        )

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker

        if df.loc[mask, "adjusted_close"].notna().any():
            continue

        logger.warning(
            "%s: sem adjusted_close disponível. "
            "Usando close_price nos gráficos de preço.",
            ticker,
        )
        df.loc[mask, "adjusted_close"] = df.loc[mask, "close_price"]

    return df


# ============================================================
# VALIDAÇÃO DE DADOS
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    requested: tuple[str, ...],
    min_observations: int = MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """
    Remove ativos com poucos dados e registra um resumo de qualidade.

    Levanta AnalysisError se nenhum ativo tiver dados suficientes.
    """

    missing = set(requested) - set(df["ticker"].unique())

    if missing:
        logger.warning(
            "Ativos sem dados em analytical_data: %s",
            ", ".join(sorted(missing)),
        )

    valid_tickers: list[str] = []

    for ticker, group in df.groupby("ticker", sort=True):
        valid_returns = int(group["return_1d"].notna().sum())

        if valid_returns < min_observations:
            logger.warning(
                "%s ignorado: apenas %s retornos válidos "
                "(mínimo: %s).",
                ticker,
                valid_returns,
                min_observations,
            )
            continue

        valid_tickers.append(str(ticker))

    if not valid_tickers:
        raise AnalysisError(
            "Nenhum ativo possui dados suficientes para a análise."
        )

    result = df[df["ticker"].isin(valid_tickers)].copy()

    # Registra a contagem de valores ausentes nas colunas analíticas.
    missing_counts = result[list(ANALYTICAL_COLUMNS)].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        logger.info(
            "Valores ausentes (esperados nas janelas iniciais e no "
            "target da última linha): %s",
            ", ".join(
                f"{column}={count}"
                for column, count in missing_counts.items()
            ),
        )

    return result.reset_index(drop=True)


# ============================================================
# ANÁLISE DESCRITIVA
# ============================================================

def _max_drawdown(prices: pd.Series) -> float:
    """Maior queda percentual a partir de um pico anterior."""

    prices = prices.dropna()

    if prices.empty:
        return float("nan")

    drawdown = prices / prices.cummax() - 1.0

    return float(drawdown.min())


def compute_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas descritivas por ativo.

    Observação: `return_1d` vem do ETL e é calculado sobre
    `close_price` (sem ajuste por dividendos/splits).
    O retorno acumulado e o drawdown usam o preço ajustado.
    """

    records: list[dict[str, object]] = []

    for ticker, group in df.groupby("ticker", sort=True):
        group = group.sort_values("price_date")

        returns = group["return_1d"].dropna()
        volume = group["volume"].dropna()
        adjusted = group["adjusted_close"].dropna()

        std_daily = float(returns.std())
        mean_daily = float(returns.mean())

        if len(adjusted) >= 2 and adjusted.iloc[0] > 0:
            cumulative = float(adjusted.iloc[-1] / adjusted.iloc[0] - 1)
        else:
            cumulative = float("nan")

        records.append(
            {
                "ticker": ticker,
                "start_date": group["price_date"].min().date(),
                "end_date": group["price_date"].max().date(),
                "observations": len(returns),
                "cumulative_return": cumulative,
                "mean_return_1d": mean_daily,
                "median_return_1d": float(returns.median()),
                "std_return_1d": std_daily,
                "annualized_return": mean_daily * TRADING_DAYS_PER_YEAR,
                "annualized_volatility": (
                    std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
                ),
                "min_return_1d": float(returns.min()),
                "max_return_1d": float(returns.max()),
                "var_95_1d": float(returns.quantile(0.05)),
                "skewness": float(returns.skew()),
                "excess_kurtosis": float(returns.kurt()),
                "max_drawdown": _max_drawdown(adjusted),
                "mean_volatility_20d": float(
                    group["volatility_20d"].mean()
                ),
                "mean_volume": float(volume.mean()),
                "median_volume": float(volume.median()),
                "max_volume": float(volume.max()),
                "zero_volume_days": int((volume == 0).sum()),
            }
        )

    return pd.DataFrame(records).set_index("ticker")


def log_descriptive_statistics(stats: pd.DataFrame) -> None:
    """Exibe as estatísticas descritivas no log/console."""

    for ticker, row in stats.iterrows():
        logger.info("=" * 60)
        logger.info("ESTATÍSTICAS DESCRITIVAS - %s", ticker)
        logger.info("=" * 60)

        for column, label, fmt in STAT_FIELDS:
            value = row[column]

            text = "n/d" if pd.isna(value) else fmt.format(value)

            logger.info("%-36s %s", label, text)

    logger.info("=" * 60)


# ============================================================
# PLOTAGEM
# ============================================================

def configure_plot_style() -> None:
    """Define um estilo visual limpo e consistente para os gráficos."""

    sns.set_theme(style="whitegrid", context="notebook")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _slug(text: str) -> str:
    """Converte um ticker em nome de arquivo seguro (HEIA.AS -> HEIA_AS)."""

    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> Path:
    """Salva a figura em disco e sempre libera a memória."""

    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    except OSError as exc:
        raise AnalysisError(
            f"Falha ao salvar o gráfico em {path}: {exc}"
        ) from exc
    finally:
        plt.close(fig)

    logger.info("Gráfico salvo: %s", path)

    return path


def _currency_of(group: pd.DataFrame) -> str:
    """Moeda do ativo (ex.: EUR) ou string vazia quando ausente."""

    values = group["currency"].dropna()

    return str(values.iloc[0]) if not values.empty else ""


# ============================================================
# FUNÇÕES DE PLOTAGEM
# ============================================================

def plot_adjusted_price_history(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """
    Histórico do fechamento ajustado, um painel por ativo.

    Cada ativo tem seu próprio eixo Y porque as moedas diferem
    (HEIA.AS em EUR, HEINY em USD).
    """

    groups = list(df.groupby("ticker", sort=True))

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(12, 4.2 * len(groups)),
        sharex=True,
        squeeze=False,
    )

    palette = sns.color_palette("deep", len(groups))

    for ax, (ticker, group), color in zip(
        axes.ravel(), groups, palette
    ):
        series = group.set_index("price_date")["adjusted_close"].dropna()

        ax.plot(series.index, series.values, color=color, linewidth=1.5)
        ax.fill_between(
            series.index,
            series.values,
            series.min(),
            color=color,
            alpha=0.08,
        )

        currency = _currency_of(group)
        ax.set_title(f"{ticker} - Fechamento ajustado")
        ax.set_ylabel(f"Preço ({currency})" if currency else "Preço")

    axes.ravel()[-1].set_xlabel("Data")
    fig.suptitle(
        "Heineken - Histórico de preços de fechamento ajustados",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )
    fig.autofmt_xdate()
    fig.tight_layout()

    return [
        _save_figure(fig, output_dir / "price_history_adjusted.png", dpi)
    ]


def plot_rebased_price_history(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """
    Compara os ativos em base 100 na primeira data em comum.

    Útil porque os preços estão em moedas diferentes; a base 100
    torna a performance relativa diretamente comparável
    (sem conversão cambial).
    """

    pivot = df.pivot_table(
        index="price_date",
        columns="ticker",
        values="adjusted_close",
    ).dropna()

    if pivot.shape[1] < 2 or len(pivot) < 2:
        logger.warning(
            "Gráfico base 100 ignorado: é necessário mais de um ativo "
            "com datas em comum."
        )
        return []

    rebased = pivot / pivot.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(12, 5.5))

    for ticker in rebased.columns:
        ax.plot(
            rebased.index,
            rebased[ticker],
            linewidth=1.6,
            label=str(ticker),
        )

    ax.axhline(100, color="gray", linestyle="--", linewidth=1)
    ax.set_title(
        "Performance relativa dos ativos (base 100, preço ajustado)"
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Índice (base 100)")
    ax.legend(title="Ativo", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()

    return [
        _save_figure(fig, output_dir / "price_history_rebased.png", dpi)
    ]


def plot_return_distribution(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """Histograma + KDE dos retornos diários, um painel por ativo."""

    groups = list(df.groupby("ticker", sort=True))

    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(7 * len(groups), 5),
        squeeze=False,
    )

    palette = sns.color_palette("deep", len(groups))

    for ax, (ticker, group), color in zip(
        axes.ravel(), groups, palette
    ):
        returns = group["return_1d"].dropna()

        sns.histplot(
            returns,
            bins=60,
            kde=True,
            stat="density",
            color=color,
            edgecolor="white",
            ax=ax,
        )

        mean = returns.mean()

        ax.axvline(
            mean,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Média: {mean:+.3%}",
        )
        ax.axvline(
            returns.quantile(0.05),
            color="firebrick",
            linestyle=":",
            linewidth=1.4,
            label=f"VaR 95%: {returns.quantile(0.05):+.2%}",
        )

        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(
            f"{ticker}  (σ = {returns.std():.2%}, "
            f"assimetria = {returns.skew():.2f})"
        )
        ax.set_xlabel("Retorno diário")
        ax.set_ylabel("Densidade")
        ax.legend(frameon=False)

    fig.suptitle(
        "Distribuição dos retornos diários",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()

    return [
        _save_figure(fig, output_dir / "return_distribution.png", dpi)
    ]


def plot_correlation_matrices(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    """
    Matriz de correlação (Pearson) entre as features, o preço e o
    target, uma figura por ativo.

    Não é feita uma matriz conjunta porque HEIA.AS (EUR) e HEINY (USD)
    têm escalas de preço diferentes.
    """

    saved: list[Path] = []

    for ticker, group in df.groupby("ticker", sort=True):
        data = group[list(CORRELATION_COLUMNS)].dropna(axis=1, how="all")
        data = data.dropna()

        if len(data) < 2 or data.shape[1] < 2:
            logger.warning(
                "%s: dados insuficientes para a matriz de correlação.",
                ticker,
            )
            continue

        corr = data.corr(method="pearson")

        # Mostra apenas o triângulo inferior (a matriz é simétrica).
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(12, 9.5))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlação de Pearson"},
            ax=ax,
        )

        ax.set_title(
            f"{ticker} - Correlação entre features e preço "
            f"(n = {len(data):,})"
        )
        ax.grid(False)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()

        path = output_dir / f"correlation_matrix_{_slug(str(ticker))}.png"
        saved.append(_save_figure(fig, path, dpi))

    return saved


# ============================================================
# ORQUESTRAÇÃO DE PLOTAGEM
# ============================================================

PlotFunction = Callable[[pd.DataFrame, Path, int], list[Path]]

PLOT_STEPS: tuple[tuple[str, PlotFunction], ...] = (
    ("histórico de preço ajustado", plot_adjusted_price_history),
    ("histórico base 100", plot_rebased_price_history),
    ("distribuição dos retornos", plot_return_distribution),
    ("matriz de correlação", plot_correlation_matrices),
)


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
) -> list[Path]:
    """
    Executa todos os gráficos. Uma falha em um gráfico não impede
    os demais; se nenhum for gerado, levanta AnalysisError.
    """

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AnalysisError(
            f"Não foi possível criar o diretório {output_dir}: {exc}"
        ) from exc

    configure_plot_style()

    generated: list[Path] = []
    failures = 0

    for name, plot_function in PLOT_STEPS:
        logger.info("Gerando gráfico: %s", name)

        try:
            generated.extend(plot_function(df, output_dir, dpi))
        except Exception as exc:
            failures += 1
            logger.exception(
                "Falha ao gerar o gráfico '%s': %s",
                name,
                exc,
            )

    if not generated:
        raise AnalysisError("Nenhum gráfico foi gerado.")

    if failures:
        logger.warning(
            "%s gráfico(s) falharam; %s gerado(s) com sucesso.",
            failures,
            len(generated),
        )

    return generated


def run_analysis(
    settings: Settings,
    tickers: tuple[str, ...] | None = None,
    output_dir: Path = FIGURES_DIR,
    dpi: int = 150,
) -> list[Path]:
    """
    Executa a EDA completa e devolve os caminhos dos gráficos gerados.

    Pode ser chamada diretamente por outros módulos do projeto:

        from analysis import run_analysis
        run_analysis(load_settings())
    """

    selected = tickers or settings.assets

    db = Database(settings)

    try:
        db.connect()

        df = load_analytical_frame(db, selected)
        df = attach_adjusted_close(db, df, selected)

    finally:
        db.close()

    df = validate_dataset(df, selected)

    stats = compute_descriptive_statistics(df)
    log_descriptive_statistics(stats)

    return generate_all_plots(df, output_dir, dpi)


# ============================================================
# INTERFACE DE LINHA DE COMANDO (CLI)
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Análise exploratória (EDA) dos ativos da Heineken."
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Tickers a analisar (padrão: variável ASSETS do .env).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Pasta de saída dos gráficos (padrão: reports/figures).",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolução dos gráficos em DPI (padrão: 150).",
    )

    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        args = parse_arguments()
        settings = load_settings()

        tickers = (
            tuple(ticker.strip().upper() for ticker in args.tickers)
            if args.tickers
            else None
        )

        logger.info("==============================================")
        logger.info("HEINEKEN ANALYTICS - ANÁLISE EXPLORATÓRIA (EDA)")
        logger.info(
            "Ativos: %s",
            ", ".join(tickers or settings.assets),
        )
        logger.info("==============================================")

        figures = run_analysis(
            settings,
            tickers=tickers,
            output_dir=args.output_dir,
            dpi=args.dpi,
        )

        logger.info(
            "EDA concluída. %s gráfico(s) salvos em %s",
            len(figures),
            args.output_dir,
        )

        return 0

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
        return 130

    except (AnalysisError, MySQLError, ConnectionError) as exc:
        logger.error("Falha na análise: %s", exc)
        return 1

    except Exception as exc:
        logger.exception("Execução encerrada com erro: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
