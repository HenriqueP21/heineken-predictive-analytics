from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

import mysql.connector
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from mysql.connector import Error as MySQLError
from plotly.subplots import make_subplots

from main import Settings, load_settings

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. CONFIGURAÇÃO INICIAL
# ============================================================

# Deve ser a primeira chamada Streamlit do script.
st.set_page_config(
    page_title="Heineken Analytics",
    page_icon="🍺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = 300
CONNECT_TIMEOUT_SECONDS = 5
TRADING_DAYS_PER_YEAR = 252

# Paleta (cores explícitas funcionam bem em tema claro e escuro).
COLOR_PRICE = "#00A651"
COLOR_UP = "#00A651"
COLOR_DOWN = "#E74C3C"
COLOR_PREDICTED = "#F5A623"
COLOR_NEUTRAL = "#7F8C8D"
SMA_COLORS = {5: "#F5A623", 10: "#4A90D9", 20: "#9B59B6"}

CURRENCY_SYMBOLS = {"EUR": "€", "USD": "US$", "BRL": "R$"}

# Períodos pré-definidos para o filtro de datas (em dias corridos).
PERIOD_PRESETS: dict[str, int | None] = {
    "1 mês": 30,
    "3 meses": 91,
    "6 meses": 182,
    "1 ano": 365,
    "3 anos": 1096,
    "Máximo": None,
}
DEFAULT_PRESET = "1 ano"
CUSTOM_PERIOD_LABEL = "Personalizado"

PRICE_COLUMNS: tuple[str, ...] = (
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "adjusted_close",
    "volume",
)

FEATURE_COLUMNS: tuple[str, ...] = (
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
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "prediction_date",
    "target_date",
    "actual_value",
    "predicted_value",
    "model_name",
    "model_version",
    "mae",
    "rmse",
    "r2",
    "created_at",
)


# ============================================================
# 2. ERROS PERSONALIZADOS
# ============================================================

class DashboardError(Exception):
    """Erro tratado pelo dashboard, com mensagem amigável."""

    title = "Erro no dashboard"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class ConfigurationError(DashboardError):
    title = "Configuração incompleta"


class DatabaseUnavailableError(DashboardError):
    title = "Banco de dados indisponível"


class DataQueryError(DashboardError):
    title = "Falha ao consultar os dados"


def _build_logger() -> logging.Logger:
    """Logger com arquivo próprio (logs/dashboard.log) e console."""

    log = logging.getLogger("heineken_dashboard")

    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    log.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "dashboard.log",
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
# 3. FORMATAÇÃO (pt-BR)
# ============================================================

def format_number(value: float | None, decimals: int = 2) -> str:
    """1234.5 -> '1.234,50'."""

    if value is None or pd.isna(value):
        return "—"

    text = f"{value:,.{decimals}f}"

    return text.replace(",", "§").replace(".", ",").replace("§", ".")


def format_percent(
    value: float | None,
    decimals: int = 2,
    signed: bool = False,
) -> str:
    """0.0123 -> '1,23%' (ou '+1,23%' com signed=True)."""

    if value is None or pd.isna(value):
        return "—"

    text = format_number(value * 100, decimals) + "%"

    return f"+{text}" if signed and value > 0 else text


def format_price(value: float | None, currency: str) -> str:
    if value is None or pd.isna(value):
        return "—"

    symbol = CURRENCY_SYMBOLS.get(currency, currency)

    return f"{symbol} {format_number(value)}".strip()


def format_date(value: date | pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "—"

    return value.strftime("%d/%m/%Y")


# ============================================================
# 4. ACESSO A DADOS (MySQL + CACHE)
# ============================================================

@dataclass(frozen=True)
class AssetInfo:
    id: int
    ticker: str
    name: str
    exchange: str
    currency: str


@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    """Lê as credenciais do .env (mesma lógica do main.py)."""

    try:
        return load_settings()
    except RuntimeError as exc:
        raise ConfigurationError(
            str(exc),
            "Copie .env.example para .env e preencha as credenciais.",
        ) from exc


@contextmanager
def db_connection() -> Iterator[mysql.connector.MySQLConnection]:
    """
    Abre uma conexão curta e SOMENTE LEITURA.

    Conexões curtas funcionam bem com o modelo de reexecução do
    Streamlit: as consultas ficam em cache e não mantemos conexões
    abertas (que poderiam expirar no servidor).
    """

    settings = get_settings()

    try:
        connection = mysql.connector.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            database=settings.mysql_database,
            user=settings.mysql_user,
            password=settings.mysql_password,
            connection_timeout=CONNECT_TIMEOUT_SECONDS,
        )
    except MySQLError as exc:
        raise DatabaseUnavailableError(
            "Não foi possível conectar ao MySQL em "
            f"{settings.mysql_host}:{settings.mysql_port}/"
            f"{settings.mysql_database}.",
            str(exc),
        ) from exc

    try:
        cursor = connection.cursor()
        try:
            # Garantia extra: o dashboard nunca escreve no banco.
            cursor.execute("SET SESSION TRANSACTION READ ONLY")
        finally:
            cursor.close()

        yield connection
    finally:
        connection.close()


def run_query(
    query: str,
    params: tuple,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Executa uma consulta e devolve um DataFrame (com colunas fixas)."""

    started = time.perf_counter()

    with db_connection() as connection:
        cursor = connection.cursor(dictionary=True)

        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        except MySQLError as exc:
            raise DataQueryError(
                "A consulta ao banco falhou. Verifique se o schema "
                "(schema.sql) foi aplicado.",
                str(exc),
            ) from exc
        finally:
            cursor.close()

    logger.info(
        "Consulta concluída: %s linhas em %.0f ms.",
        len(rows),
        (time.perf_counter() - started) * 1000,
    )

    return pd.DataFrame(rows, columns=list(columns))


def _to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Carregando ativos...")
def load_assets() -> list[AssetInfo]:
    df = run_query(
        """
        SELECT id, ticker, name, exchange, currency
        FROM assets
        WHERE active = TRUE
        ORDER BY ticker
        """,
        (),
        ("id", "ticker", "name", "exchange", "currency"),
    )

    return [
        AssetInfo(
            id=int(row.id),
            ticker=str(row.ticker),
            name=str(row.name),
            exchange=str(row.exchange or "—"),
            currency=str(row.currency or ""),
        )
        for row in df.itertuples(index=False)
    ]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Carregando preços...")
def load_asset_data(asset_id: int) -> pd.DataFrame:
    """
    Série diária do ativo: `market_prices` (OHLC, ajustado, volume)
    combinada com `analytical_data` (retornos, SMAs, volatilidades).
    """

    columns = ("price_date",) + PRICE_COLUMNS + FEATURE_COLUMNS

    price_select = ", ".join(f"mp.{column}" for column in PRICE_COLUMNS)
    feature_select = ", ".join(f"ad.{column}" for column in FEATURE_COLUMNS)

    # Consulta LEFT JOIN: nem todos os dias têm dados analíticos (ex: feriados).
    df = run_query(
        f"""
        SELECT
            mp.price_date,
            {price_select},
            {feature_select}
        FROM market_prices AS mp
        LEFT JOIN analytical_data AS ad
            ON ad.asset_id = mp.asset_id
           AND ad.price_date = mp.price_date
        WHERE mp.asset_id = %s
        ORDER BY mp.price_date
        """,
        (asset_id,),
        columns,
    )

    df["price_date"] = pd.to_datetime(df["price_date"])
    df = _to_numeric(df, columns[1:])

    # Fechamento ajustado com alternativa ao fechamento simples.
    df["adj_close"] = df["adjusted_close"].fillna(df["close_price"])

    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner="Carregando previsões...")
def load_predictions(asset_id: int) -> pd.DataFrame:
    df = run_query(
        f"""
        SELECT {", ".join(PREDICTION_COLUMNS)}
        FROM predictions
        WHERE asset_id = %s
        ORDER BY prediction_date
        """,
        (asset_id,),
        PREDICTION_COLUMNS,
    )

    for column in ("prediction_date", "target_date", "created_at"):
        df[column] = pd.to_datetime(df[column])

    return _to_numeric(
        df,
        ("actual_value", "predicted_value", "mae", "rmse", "r2"),
    )


# ============================================================
# 5. LÓGICA DE NEGÓCIO
# ============================================================

@dataclass(frozen=True)
class ModelVersion:
    name: str
    version: str
    label: str


@dataclass(frozen=True)
class Filters:
    start: pd.Timestamp
    end: pd.Timestamp
    model: ModelVersion | None


@dataclass(frozen=True)
class Kpis:
    last_price: float | None
    last_date: pd.Timestamp | None
    price_change: float | None
    cumulative_return: float | None
    volatility_annualized: float | None
    volatility_vs_period: float | None
    forecast_value: float | None
    forecast_target_date: pd.Timestamp | None
    forecast_change: float | None
    forecast_pending: bool
    forecast_actual: float | None
    forecast_version: str | None


@dataclass(frozen=True)
class ForecastEvaluation:
    observations: int
    mae: float
    rmse: float
    mape: float
    baseline_mae: float | None
    improvement_vs_baseline: float | None


def list_model_versions(predictions: pd.DataFrame) -> list[ModelVersion]:
    """Versões de modelo presentes em `predictions`, da mais recente."""

    if predictions.empty:
        return []

    grouped = (
        predictions.groupby(["model_name", "model_version"], dropna=False)[
            "created_at"
        ]
        .max()
        .sort_values(ascending=False)
    )

    return [
        ModelVersion(
            name=str(name),
            version=str(version),
            label=f"{name} · v{version}",
        )
        for (name, version) in grouped.index
    ]


def enrich_predictions(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complementa as previsões com dados de mercado.

    - `close_at_prediction`: fechamento do dia da previsão (usado no
      baseline "amanhã = hoje").
    - `actual_value`: a previsão do próximo pregão é gravada sem valor
      real; quando o pregão alvo já existe em `market_prices`, o valor
      real é preenchido aqui.
    - `error`, `abs_pct_error`: erro da previsão (previsto - real).
    """

    result = predictions.copy()

    close_by_date = data.drop_duplicates("price_date").set_index(
        "price_date"
    )["close_price"]

    result["close_at_prediction"] = result["prediction_date"].map(
        close_by_date
    )
    result["actual_value"] = result["actual_value"].fillna(
        result["target_date"].map(close_by_date)
    )

    result["is_pending"] = result["actual_value"].isna()
    result["error"] = result["predicted_value"] - result["actual_value"]
    result["abs_pct_error"] = result["error"].abs() / result["actual_value"]

    return result


def select_model_predictions(
    predictions: pd.DataFrame,
    model: ModelVersion | None,
) -> pd.DataFrame:
    if model is None or predictions.empty:
        return predictions.iloc[0:0]

    mask = (predictions["model_name"] == model.name) & (
        predictions["model_version"].astype(str) == model.version
    )

    return predictions[mask].sort_values("prediction_date")


def filter_by_period(
    df: pd.DataFrame,
    column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    mask = (df[column] >= start) & (df[column] <= end)

    return df[mask]


def compute_kpis(
    view: pd.DataFrame,
    model_predictions: pd.DataFrame,
) -> Kpis:
    """KPIs do topo da página."""

    last = view.iloc[-1]

    price_change = None

    if len(view) >= 2:
        previous = view["close_price"].iloc[-2]

        if previous and not pd.isna(previous):
            price_change = float(last["close_price"] / previous - 1)

    adjusted = view["adj_close"].dropna()
    cumulative = None

    if len(adjusted) >= 2 and adjusted.iloc[0] > 0:
        cumulative = float(adjusted.iloc[-1] / adjusted.iloc[0] - 1)

    volatility = view["volatility_20d"].dropna()
    vol_annualized = None
    vol_vs_period = None

    if not volatility.empty:
        factor = np.sqrt(TRADING_DAYS_PER_YEAR)
        vol_annualized = float(volatility.iloc[-1] * factor)
        vol_vs_period = float((volatility.iloc[-1] - volatility.mean()) * factor)

    forecast_value = None
    forecast_target = None
    forecast_change = None
    forecast_pending = False
    forecast_actual = None
    forecast_version = None

    if not model_predictions.empty:
        row = model_predictions.sort_values("prediction_date").iloc[-1]

        forecast_value = float(row["predicted_value"])
        forecast_target = row["target_date"]
        forecast_pending = bool(row["is_pending"])
        forecast_version = str(row["model_version"])

        if not pd.isna(row["actual_value"]):
            forecast_actual = float(row["actual_value"])

        base = row["close_at_prediction"]

        if pd.isna(base):
            base = last["close_price"]

        if base and not pd.isna(base):
            forecast_change = float(forecast_value / base - 1)

    return Kpis(
        last_price=float(last["close_price"]),
        last_date=last["price_date"],
        price_change=price_change,
        cumulative_return=cumulative,
        volatility_annualized=vol_annualized,
        volatility_vs_period=vol_vs_period,
        forecast_value=forecast_value,
        forecast_target_date=forecast_target,
        forecast_change=forecast_change,
        forecast_pending=forecast_pending,
        forecast_actual=forecast_actual,
        forecast_version=forecast_version,
    )


def evaluate_forecasts(predictions: pd.DataFrame) -> ForecastEvaluation | None:
    """Métricas recalculadas sobre as previsões já realizadas no período."""

    evaluated = predictions[~predictions["is_pending"]].dropna(
        subset=["error"]
    )

    if evaluated.empty:
        return None

    errors = evaluated["error"]
    mae = float(errors.abs().mean())

    baseline_mae = None
    improvement = None

    baseline_errors = (
        evaluated["close_at_prediction"] - evaluated["actual_value"]
    ).dropna()

    if not baseline_errors.empty:
        baseline_mae = float(baseline_errors.abs().mean())

        if baseline_mae > 0:
            improvement = 1 - mae / baseline_mae

    return ForecastEvaluation(
        observations=len(evaluated),
        mae=mae,
        rmse=float(np.sqrt((errors**2).mean())),
        mape=float(evaluated["abs_pct_error"].mean()),
        baseline_mae=baseline_mae,
        improvement_vs_baseline=improvement,
    )


def compute_drawdown(prices: pd.Series) -> pd.Series:
    prices = prices.dropna()

    return prices / prices.cummax() - 1


def to_csv_bytes(df: pd.DataFrame, brazilian_format: bool) -> bytes:
    """
    CSV em UTF-8 com BOM (abre corretamente no Excel).

    Formato Brasil: separador ';' e decimal ','.
    """

    export = df.copy()

    for column in export.select_dtypes(include=["datetime64[ns]"]).columns:
        export[column] = export[column].dt.strftime("%Y-%m-%d")

    sep, decimal = (";", ",") if brazilian_format else (",", ".")

    return export.to_csv(index=False, sep=sep, decimal=decimal).encode(
        "utf-8-sig"
    )


# ============================================================
# 6. GRÁFICOS (PLOTLY)
# ============================================================

def apply_layout(fig: go.Figure, height: int = 520) -> go.Figure:
    """Layout comum a todos os gráficos."""

    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        # Separadores no padrão pt-BR: decimal ',' e milhar '.'.
        separators=",.",
    )

    return fig


def build_price_chart(
    view: pd.DataFrame,
    chart_type: str,
    sma_windows: Sequence[int],
    currency: str,
) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.76, 0.24],
        vertical_spacing=0.03,
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=view["price_date"],
                open=view["open_price"],
                high=view["high_price"],
                low=view["low_price"],
                close=view["close_price"],
                increasing_line_color=COLOR_UP,
                decreasing_line_color=COLOR_DOWN,
                name="Preço",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=view["price_date"],
                y=view["close_price"],
                mode="lines",
                name="Fechamento",
                line={"color": COLOR_PRICE, "width": 2},
            ),
            row=1,
            col=1,
        )

    for window in sma_windows:
        column = f"sma_{window}"

        if column not in view or view[column].notna().sum() == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=view["price_date"],
                y=view[column],
                mode="lines",
                name=f"SMA {window}",
                line={"color": SMA_COLORS.get(window), "width": 1.4},
            ),
            row=1,
            col=1,
        )

    direction = view["close_price"].diff().fillna(0) >= 0
    volume_colors = np.where(direction, COLOR_UP, COLOR_DOWN).tolist()

    fig.add_trace(
        go.Bar(
            x=view["price_date"],
            y=view["volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.55,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text=f"Preço ({currency})" if currency else "Preço",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Remove fins de semana do eixo (evita "buracos" no gráfico).
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    fig.update_layout(xaxis_rangeslider_visible=False)

    return apply_layout(fig, height=620)


def build_volatility_chart(
    view: pd.DataFrame,
    annualize: bool,
) -> go.Figure:
    factor = np.sqrt(TRADING_DAYS_PER_YEAR) if annualize else 1.0

    fig = go.Figure()

    for column, label, color in (
        ("volatility_5d", "Volatilidade 5d", COLOR_PREDICTED),
        ("volatility_20d", "Volatilidade 20d", COLOR_PRICE),
    ):
        fig.add_trace(
            go.Scatter(
                x=view["price_date"],
                y=view[column] * factor,
                mode="lines",
                name=label,
                line={"color": color, "width": 1.8},
            )
        )

    fig.update_yaxes(
        title_text="Volatilidade anualizada" if annualize else "Volatilidade diária",
        tickformat=".1%",
    )
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])

    return apply_layout(fig, height=380)


def build_return_histogram(returns: pd.Series) -> go.Figure:
    returns = returns.dropna()

    fig = go.Figure(
        go.Histogram(
            x=returns,
            nbinsx=50,
            marker_color=COLOR_PRICE,
            opacity=0.85,
            name="Retornos diários",
        )
    )

    if len(returns) >= 2:
        var_95 = float(returns.quantile(0.05))

        fig.add_vline(
            x=float(returns.mean()),
            line_dash="dash",
            line_color=COLOR_NEUTRAL,
            annotation_text="Média",
            annotation_position="top",
        )
        fig.add_vline(
            x=var_95,
            line_dash="dot",
            line_color=COLOR_DOWN,
            annotation_text="VaR 95%",
            annotation_position="top left",
        )

    fig.update_xaxes(title_text="Retorno diário", tickformat=".1%")
    fig.update_yaxes(title_text="Frequência")
    fig.update_layout(showlegend=False)

    return apply_layout(fig, height=360).update_layout(hovermode="closest")


def build_drawdown_chart(adj_close: pd.Series, dates: pd.Series) -> go.Figure:
    drawdown = compute_drawdown(adj_close)

    fig = go.Figure(
        go.Scatter(
            x=dates.loc[drawdown.index],
            y=drawdown,
            mode="lines",
            fill="tozeroy",
            line={"color": COLOR_DOWN, "width": 1.4},
            name="Drawdown",
        )
    )

    fig.update_yaxes(title_text="Queda desde o pico", tickformat=".0%")
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    fig.update_layout(showlegend=False)

    return apply_layout(fig, height=360)


def build_forecast_chart(
    predictions: pd.DataFrame,
    currency: str,
) -> go.Figure:
    evaluated = predictions[~predictions["is_pending"]]
    pending = predictions[predictions["is_pending"]]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.05,
    )

    fig.add_trace(
        go.Scatter(
            x=evaluated["target_date"],
            y=evaluated["actual_value"],
            mode="lines",
            name="Preço real",
            line={"color": COLOR_PRICE, "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=evaluated["target_date"],
            y=evaluated["predicted_value"],
            mode="lines",
            name="Previsão Random Forest",
            line={"color": COLOR_PREDICTED, "width": 1.8, "dash": "dot"},
        ),
        row=1,
        col=1,
    )

    if not pending.empty:
        fig.add_trace(
            go.Scatter(
                x=pending["target_date"],
                y=pending["predicted_value"],
                mode="markers",
                name="Próxima previsão",
                marker={
                    "symbol": "star",
                    "size": 15,
                    "color": COLOR_PREDICTED,
                    "line": {"color": "white", "width": 1},
                },
            ),
            row=1,
            col=1,
        )

    errors = evaluated["error"]

    fig.add_trace(
        go.Bar(
            x=evaluated["target_date"],
            y=errors,
            name="Erro (previsto − real)",
            marker_color=np.where(errors >= 0, COLOR_PREDICTED, "#4A90D9").tolist(),
            opacity=0.8,
        ),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text=f"Preço ({currency})" if currency else "Preço",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Erro", row=2, col=1)
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])

    return apply_layout(fig, height=600)


# ============================================================
# 7. INTERFACE STREAMLIT (UI)
# ============================================================

def inject_styles() -> None:
    """Ajustes finos de CSS (compatíveis com tema claro e escuro)."""

    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
            h1 { letter-spacing: -0.5px; margin-bottom: 0; }
            [data-testid="stMetricLabel"] p { font-size: 0.85rem; opacity: 0.75; }
            [data-testid="stMetricValue"] { font-size: 1.7rem; font-weight: 700; }
            footer { visibility: hidden; }
            @media (max-width: 640px) {
                [data-testid="stMetricValue"] { font-size: 1.4rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title("🍺 Heineken Analytics")
    st.caption(
        "Análise financeira preditiva · Grupo Heineken · "
        "Dados de mercado, indicadores e previsões do Random Forest"
    )


def render_error(error: DashboardError) -> None:
    """Mensagem amigável para falhas de banco/configuração."""

    st.error(f"**{error.title}**\n\n{error.message}")

    with st.expander("Como resolver", expanded=True):
        if isinstance(error, ConfigurationError):
            st.markdown(
                "- Copie `.env.example` para `.env` na pasta do projeto.\n"
                "- Preencha `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, "
                "`MYSQL_USER` e `MYSQL_PASSWORD`.\n"
                "- Reinicie o dashboard."
            )
        elif isinstance(error, DatabaseUnavailableError):
            st.markdown(
                "- Confirme que o servidor MySQL está em execução.\n"
                "- Verifique host, porta, usuário e senha no `.env`.\n"
                "- Confirme que o banco `heineken_analytics` existe "
                "(`schema.sql`).\n"
                "- Verifique firewall/rede, se o banco for remoto."
            )
        else:
            st.markdown(
                "- Aplique o `schema.sql` no banco.\n"
                "- Execute o pipeline: `python main.py`.\n"
                "- Consulte `logs/dashboard.log` para mais detalhes."
            )

    if error.detail:
        with st.expander("Detalhes técnicos"):
            st.code(error.detail)

    if st.button("🔄 Tentar novamente", key="retry_button"):
        st.cache_data.clear()
        st.rerun()


def render_asset_selector(assets: list[AssetInfo]) -> AssetInfo:
    st.sidebar.header("Filtros")

    by_ticker = {asset.ticker: asset for asset in assets}

    ticker = st.sidebar.selectbox(
        "Ativo",
        options=list(by_ticker),
        format_func=lambda t: f"{t} · {by_ticker[t].name}",
        key="asset_ticker",
    )

    return by_ticker[ticker]


def render_filters(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Filters | None:
    """Período e versão do modelo. Retorna None se o filtro estiver incompleto."""

    data_min = data["price_date"].min()
    data_max = data["price_date"].max()

    preset = st.sidebar.selectbox(
        "Período",
        options=[*PERIOD_PRESETS, CUSTOM_PERIOD_LABEL],
        index=list(PERIOD_PRESETS).index(DEFAULT_PRESET),
        key="period_preset",
    )

    if preset == CUSTOM_PERIOD_LABEL:
        default_start = max(data_min, data_max - pd.Timedelta(days=365))

        selected = st.sidebar.date_input(
            "Intervalo de datas",
            value=(default_start.date(), data_max.date()),
            min_value=data_min.date(),
            max_value=data_max.date(),
            format="DD/MM/YYYY",
            key="custom_range",
        )

        if not isinstance(selected, tuple) or len(selected) != 2:
            st.sidebar.info("Selecione a data inicial e a data final.")
            return None

        start, end = (pd.Timestamp(item) for item in selected)
    else:
        days = PERIOD_PRESETS[preset]
        end = data_max
        start = data_min if days is None else max(
            data_min, data_max - pd.Timedelta(days=days)
        )

    versions = list_model_versions(predictions)
    model: ModelVersion | None = None

    if versions:
        by_label = {version.label: version for version in versions}

        label = st.sidebar.selectbox(
            "Modelo (previsões)",
            options=list(by_label),
            key="model_version",
        )
        model = by_label[label]
    else:
        st.sidebar.caption(
            "Sem previsões para este ativo. Execute "
            "`python model_training.py`."
        )

    st.sidebar.divider()

    if st.sidebar.button("🔄 Atualizar dados", key="refresh_button"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.caption(
        f"Dados de {format_date(data_min)} a {format_date(data_max)}. "
        f"Cache de {CACHE_TTL_SECONDS // 60} min."
    )

    return Filters(start=start, end=end, model=model)


def render_asset_header(asset: AssetInfo, view: pd.DataFrame) -> None:
    st.subheader(f"{asset.name} ({asset.ticker})")
    st.caption(
        f"{asset.exchange} · Moeda: {asset.currency or '—'} · "
        f"Período: {format_date(view['price_date'].min())} a "
        f"{format_date(view['price_date'].max())} · "
        f"{len(view):,} pregões".replace(",", ".")
    )


def render_kpis(kpis: Kpis, asset: AssetInfo) -> None:
    columns = st.columns(4)

    with columns[0]:
        with st.container(border=True):
            st.metric(
                "Preço atual",
                format_price(kpis.last_price, asset.currency),
                format_percent(kpis.price_change, signed=True)
                if kpis.price_change is not None
                else None,
                help="Último fechamento do período e variação sobre o "
                "pregão anterior.",
            )
            st.caption(f"Fechamento de {format_date(kpis.last_date)}")

    with columns[1]:
        with st.container(border=True):
            st.metric(
                "Retorno acumulado",
                format_percent(kpis.cumulative_return, signed=True),
                help="Variação do fechamento ajustado entre o primeiro e "
                "o último pregão do período selecionado.",
            )
            st.caption("No período selecionado (preço ajustado)")

    with columns[2]:
        with st.container(border=True):
            delta = None

            if kpis.volatility_vs_period is not None:
                delta = (
                    f"{format_number(kpis.volatility_vs_period * 100, 1)} p.p."
                )
                if kpis.volatility_vs_period > 0:
                    delta = f"+{delta}"

            st.metric(
                "Volatilidade (20d)",
                format_percent(kpis.volatility_annualized, 1),
                delta,
                delta_color="inverse",
                help="Desvio-padrão dos retornos diários dos últimos 20 "
                "pregões, anualizado (× √252). A variação compara com a "
                "média do período.",
            )
            st.caption("Anualizada · variação vs. média do período")

    with columns[3]:
        with st.container(border=True):
            if kpis.forecast_value is None:
                st.metric("Previsão Random Forest", "—")
                st.caption("Sem previsões disponíveis")
            else:
                st.metric(
                    "Previsão Random Forest",
                    format_price(kpis.forecast_value, asset.currency),
                    format_percent(kpis.forecast_change, signed=True)
                    if kpis.forecast_change is not None
                    else None,
                    help="Última previsão gravada em `predictions` e a "
                    "variação esperada sobre o fechamento do dia em que "
                    "foi gerada.",
                )

                caption = f"Alvo: {format_date(kpis.forecast_target_date)}"

                if not kpis.forecast_pending and kpis.forecast_actual is not None:
                    caption += (
                        " · real: "
                        f"{format_price(kpis.forecast_actual, asset.currency)}"
                    )

                st.caption(f"{caption} · v{kpis.forecast_version}")


def render_price_tab(view: pd.DataFrame, asset: AssetInfo) -> None:
    controls = st.columns([1, 2])

    with controls[0]:
        chart_type = st.radio(
            "Tipo de gráfico",
            ["Linha", "Candlestick"],
            horizontal=True,
            key="price_chart_type",
        )

    with controls[1]:
        windows = st.multiselect(
            "Médias móveis simples",
            options=[5, 10, 20],
            default=[5, 20],
            format_func=lambda w: f"SMA {w}",
            key="price_sma_windows",
        )

    st.plotly_chart(
        build_price_chart(view, chart_type, windows, asset.currency),
        key="chart_price",
        config={"displaylogo": False},
    )

    st.caption(
        "Preços e médias móveis calculados sobre o fechamento "
        "(sem ajuste por dividendos). Fins de semana ocultos no eixo."
    )


def render_volatility_tab(view: pd.DataFrame) -> None:
    returns = view["return_1d"].dropna()

    if returns.empty:
        st.info(
            "Sem retornos calculados no período. Execute o ETL "
            "(`python main.py --mode etl`)."
        )
        return

    annualize = st.toggle("Anualizar volatilidade", value=True, key="vol_annualize")

    st.markdown("##### Volatilidade móvel")
    st.plotly_chart(
        build_volatility_chart(view, annualize),
        key="chart_volatility",
        config={"displaylogo": False},
    )

    st.markdown("##### Estatísticas dos retornos diários")

    stats = st.columns(5)
    std_daily = float(returns.std()) if len(returns) > 1 else float("nan")

    cards = (
        ("Retorno médio/dia", format_percent(returns.mean(), 3, signed=True)),
        (
            "Volatilidade anual.",
            format_percent(std_daily * np.sqrt(TRADING_DAYS_PER_YEAR), 1),
        ),
        ("Melhor dia", format_percent(returns.max(), 2, signed=True)),
        ("Pior dia", format_percent(returns.min(), 2, signed=True)),
        ("VaR 95% (1 dia)", format_percent(returns.quantile(0.05), 2, signed=True)),
    )

    for column, (label, value) in zip(stats, cards):
        with column:
            with st.container(border=True):
                st.metric(label, value)

    left, right = st.columns(2)

    with left:
        st.markdown("##### Distribuição dos retornos")
        st.plotly_chart(
            build_return_histogram(returns),
            key="chart_returns_hist",
            config={"displaylogo": False},
        )

    with right:
        st.markdown("##### Drawdown (queda desde o pico)")
        st.plotly_chart(
            build_drawdown_chart(view["adj_close"], view["price_date"]),
            key="chart_drawdown",
            config={"displaylogo": False},
        )

    st.caption(
        "Retornos e volatilidades vêm do ETL e usam o fechamento sem "
        "ajuste; o drawdown usa o fechamento ajustado."
    )


def render_forecast_tab(
    asset: AssetInfo,
    model: ModelVersion | None,
    model_predictions: pd.DataFrame,
    all_model_predictions: pd.DataFrame,
) -> None:
    if model is None or all_model_predictions.empty:
        st.info(
            "Ainda não há previsões para este ativo. Execute "
            "`python model_training.py` para treinar o modelo e gravar "
            "as previsões na tabela `predictions`."
        )
        return

    if model_predictions.empty:
        first = all_model_predictions["prediction_date"].min()
        last = all_model_predictions["prediction_date"].max()

        st.warning(
            "Não há previsões no período selecionado. As previsões deste "
            f"modelo cobrem de {format_date(first)} a {format_date(last)}; "
            "amplie o período na barra lateral."
        )
        return

    evaluation = evaluate_forecasts(model_predictions)

    if evaluation is None:
        st.info("O período contém apenas a previsão do próximo pregão.")
    else:
        columns = st.columns(4)

        with columns[0]:
            with st.container(border=True):
                st.metric(
                    "MAE",
                    format_price(evaluation.mae, asset.currency),
                    help="Erro absoluto médio no período selecionado.",
                )

        with columns[1]:
            with st.container(border=True):
                st.metric(
                    "RMSE",
                    format_price(evaluation.rmse, asset.currency),
                    help="Raiz do erro quadrático médio (penaliza erros grandes).",
                )

        with columns[2]:
            with st.container(border=True):
                st.metric("MAPE", format_percent(evaluation.mape))

        with columns[3]:
            with st.container(border=True):
                improvement = evaluation.improvement_vs_baseline

                st.metric(
                    "MAE vs. baseline",
                    format_percent(improvement, 1, signed=True)
                    if improvement is not None
                    else "—",
                    help="Compara o MAE do modelo com o baseline "
                    "'amanhã = hoje'. Positivo = o modelo erra menos.",
                )

        if improvement is not None and improvement <= 0:
            st.warning(
                "No período selecionado o modelo **não supera** o baseline "
                "ingênuo (repetir o preço de hoje). Interprete as "
                "previsões com cautela."
            )

    stored = all_model_predictions.dropna(subset=["mae"]).head(1)

    if not stored.empty:
        row = stored.iloc[0]
        st.caption(
            "Métricas gravadas no treinamento (período de teste): "
            f"MAE {format_number(row['mae'], 4)} · "
            f"RMSE {format_number(row['rmse'], 4)} · "
            f"R² {format_number(row['r2'], 4)}"
        )

    st.plotly_chart(
        build_forecast_chart(model_predictions, asset.currency),
        key="chart_forecast",
        config={"displaylogo": False},
    )

    st.caption(
        "Cada ponto compara o preço real de uma data com o que o modelo "
        "previu no pregão anterior. A estrela marca a previsão do próximo "
        "pregão, ainda sem valor real. O R² sobre preços tende a ser alto "
        "mesmo para modelos fracos; use o baseline como referência."
    )


def render_data_tab(
    view: pd.DataFrame,
    model_predictions: pd.DataFrame,
    asset: AssetInfo,
    filters: Filters,
) -> None:
    dataset = st.radio(
        "Conjunto de dados",
        ["Preços e indicadores", "Previsões"],
        horizontal=True,
        key="data_dataset",
    )

    if dataset == "Preços e indicadores":
        table = view.drop(columns=["adj_close"]).sort_values(
            "price_date", ascending=False
        )
        date_columns = {
            "price_date": st.column_config.DateColumn(
                "Data", format="DD/MM/YYYY"
            )
        }
        slug = "precos"
    else:
        if model_predictions.empty:
            st.info("Sem previsões no período selecionado.")
            return

        table = model_predictions[
            [
                "prediction_date",
                "target_date",
                "actual_value",
                "predicted_value",
                "error",
                "abs_pct_error",
                "model_name",
                "model_version",
                "mae",
                "rmse",
                "r2",
            ]
        ].sort_values("prediction_date", ascending=False)
        date_columns = {
            "prediction_date": st.column_config.DateColumn(
                "Data da previsão", format="DD/MM/YYYY"
            ),
            "target_date": st.column_config.DateColumn(
                "Data alvo", format="DD/MM/YYYY"
            ),
        }
        slug = "previsoes"

    st.dataframe(
        table,
        hide_index=True,
        height=480,
        column_config=date_columns,
    )

    left, right = st.columns([2, 3])

    with left:
        csv_format = st.selectbox(
            "Formato do CSV",
            ["Internacional (vírgula)", "Excel Brasil (ponto e vírgula)"],
            key="csv_format",
        )

    with right:
        st.write("")  # Espaço para alinhar o botão à direita
        st.download_button(
            "⬇️ Exportar CSV",
            data=to_csv_bytes(
                table,
                brazilian_format=csv_format.startswith("Excel"),
            ),
            file_name=(
                f"heineken_{slug}_{asset.ticker}_"
                f"{filters.start:%Y%m%d}_{filters.end:%Y%m%d}.csv"
            ),
            mime="text/csv",
            key="download_csv",
        )

    st.caption(f"{len(table):,} registros".replace(",", "."))


def render_footer() -> None:
    st.divider()
    st.caption(
        "Projeto acadêmico. As previsões são resultado de um modelo "
        "estatístico experimental e não constituem recomendação de "
        "investimento."
    )


# ============================================================
# 8. EXECUÇÃO DO DASHBOARD
# ============================================================

def run_dashboard() -> None:
    assets = load_assets()

    if not assets:
        st.warning(
            "Nenhum ativo ativo encontrado na tabela `assets`. "
            "Aplique o `schema.sql` e execute `python main.py`."
        )
        return

    asset = render_asset_selector(assets)

    data = load_asset_data(asset.id)

    if data.empty:
        st.warning(
            f"Não há dados de mercado para {asset.ticker}. "
            "Execute a coleta (`python main.py --mode collect`)."
        )
        return

    predictions = load_predictions(asset.id)

    filters = render_filters(data, predictions)

    if filters is None:
        return

    view = filter_by_period(data, "price_date", filters.start, filters.end)

    if view.empty:
        st.warning("Não há dados no período selecionado.")
        return

    enriched = enrich_predictions(predictions, data)
    all_model_predictions = select_model_predictions(enriched, filters.model)
    period_predictions = filter_by_period(
        all_model_predictions,
        "prediction_date",
        filters.start,
        filters.end,
    )

    render_asset_header(asset, view)
    render_kpis(compute_kpis(view, all_model_predictions), asset)

    st.write("")

    tab_price, tab_volatility, tab_forecast, tab_data = st.tabs(
        [
            "📈 Preços e médias móveis",
            "📉 Volatilidade e retornos",
            "🤖 Real vs. previsão",
            "🗂️ Dados brutos",
        ]
    )

    with tab_price:
        render_price_tab(view, asset)

    with tab_volatility:
        render_volatility_tab(view)

    with tab_forecast:
        render_forecast_tab(
            asset,
            filters.model,
            period_predictions,
            all_model_predictions,
        )

    with tab_data:
        render_data_tab(view, period_predictions, asset, filters)

    render_footer()

# ============================================================
# 9. PONTO DE ENTRADA
# ============================================================

def main() -> None:
    inject_styles()
    render_header()

    try:
        run_dashboard()

    except DashboardError as exc:
        logger.error("%s: %s | %s", exc.title, exc.message, exc.detail)
        render_error(exc)

    except Exception as exc:
        logger.exception("Erro inesperado no dashboard: %s", exc)
        render_error(
            DashboardError(
                "Ocorreu um erro inesperado ao montar o dashboard.",
                f"{type(exc).__name__}: {exc}",
            )
        )


main()
