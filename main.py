from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mysql.connector
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from mysql.connector import Error

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CAMINHO, BASE E LOGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")


# ============================================================
# LOGGING (LOGGER)
# ============================================================

LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("heineken_pipeline")


# ============================================================
# CONFIGURAÇÕES
# ============================================================

@dataclass(frozen=True)
class Settings:
    mysql_host: str
    mysql_port: int
    mysql_database: str
    mysql_user: str
    mysql_password: str

    period: str
    interval: str
    request_timeout: int

    assets: tuple[str, ...]


def load_settings() -> Settings:
    required = [
        "MYSQL_HOST",
        "MYSQL_PORT",
        "MYSQL_DATABASE",
        "MYSQL_USER",
        "MYSQL_PASSWORD",
    ]

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise RuntimeError(
            "Variáveis obrigatórias ausentes no .env: "
            + ", ".join(missing)
        )

    assets_raw = os.getenv("ASSETS", "HEIA.AS,HEINY")

    assets = tuple(
        ticker.strip()
        for ticker in assets_raw.split(",")
        if ticker.strip()
    )

    if not assets:
        raise RuntimeError("Nenhum ativo foi configurado em ASSETS.")

    return Settings(
        mysql_host=os.getenv("MYSQL_HOST", "localhost"),
        mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
        mysql_database=os.getenv(
            "MYSQL_DATABASE",
            "heineken_analytics",
        ),
        mysql_user=os.getenv("MYSQL_USER", "root"),
        mysql_password=os.getenv("MYSQL_PASSWORD", ""),

        period=os.getenv("YFINANCE_PERIOD", "5y"),
        interval=os.getenv("YFINANCE_INTERVAL", "1d"),
        request_timeout=int(
            os.getenv("YFINANCE_TIMEOUT", "30")
        ),

        assets=assets,
    )


# ============================================================
# DATABASE
# ============================================================

class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection: mysql.connector.MySQLConnection | None = None

    def connect(self) -> None:
        logger.info(
            "Conectando ao MySQL %s:%s/%s",
            self.settings.mysql_host,
            self.settings.mysql_port,
            self.settings.mysql_database,
        )

        self.connection = mysql.connector.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            database=self.settings.mysql_database,
            user=self.settings.mysql_user,
            password=self.settings.mysql_password,
            autocommit=False,
        )

        if not self.connection.is_connected():
            raise ConnectionError("Não foi possível conectar ao MySQL.")

        logger.info("Conexão MySQL estabelecida.")

    def close(self) -> None:
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Conexão MySQL encerrada.")

    def cursor(self):
        if not self.connection or not self.connection.is_connected():
            raise ConnectionError("Banco de dados não conectado.")

        return self.connection.cursor(dictionary=True)

    def execute(
        self,
        query: str,
        params: tuple | list | None = None,
    ) -> int:
        if not self.connection:
            raise ConnectionError("Banco não conectado.")

        cursor = self.connection.cursor()
        try:
            cursor.execute(query, params or ())
            affected = cursor.rowcount
            self.connection.commit()
            return affected
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def executemany(
        self,
        query: str,
        rows: list[tuple],
    ) -> int:
        if not self.connection:
            raise ConnectionError("Banco não conectado.")

        if not rows:
            return 0

        cursor = self.connection.cursor()

        try:
            cursor.executemany(query, rows)
            affected = cursor.rowcount
            self.connection.commit()
            return affected
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()


# ============================================================
# PIPELINE RUN TRACKING
# ============================================================

class PipelineRun:
    def __init__(self, db: Database):
        self.db = db
        self.run_id: int | None = None
        self.started_at: datetime | None = None

    def start(self) -> int:
        self.started_at = datetime.now()

        query = """
            INSERT INTO pipeline_runs (
                execution_date,
                status,
                started_at
            )
            VALUES (%s, 'RUNNING', %s)
        """

        self.db.execute(
            query,
            (
                self.started_at,
                self.started_at,
            ),
        )

        cursor = self.db.connection.cursor()
        try:
            cursor.execute("SELECT LAST_INSERT_ID() AS id")
            row = cursor.fetchone()
            self.run_id = int(row[0])
        finally:
            cursor.close()

        logger.info("Pipeline iniciado. run_id=%s", self.run_id)
        return self.run_id

    def finish(
        self,
        status: str,
        records_collected: int = 0,
        records_processed: int = 0,
        predictions_generated: int = 0,
        error_message: str | None = None,
    ) -> None:
        if self.run_id is None:
            return

        finished_at = datetime.now()

        query = """
            UPDATE pipeline_runs
            SET
                status = %s,
                records_collected = %s,
                records_processed = %s,
                predictions_generated = %s,
                error_message = %s,
                finished_at = %s
            WHERE id = %s
        """

        self.db.execute(
            query,
            (
                status,
                records_collected,
                records_processed,
                predictions_generated,
                error_message,
                finished_at,
                self.run_id,
            ),
        )

        logger.info(
            "Pipeline finalizado. run_id=%s status=%s",
            self.run_id,
            status,
        )


# ============================================================
# ASSETS
# ============================================================

def get_asset_map(
    db: Database,
    tickers: tuple[str, ...],
) -> dict[str, int]:
    placeholders = ", ".join(["%s"] * len(tickers))

    query = f"""
        SELECT id, ticker
        FROM assets
        WHERE ticker IN ({placeholders})
          AND active = TRUE
    """

    cursor = db.cursor()

    try:
        cursor.execute(query, tickers)
        rows = cursor.fetchall()
    finally:
        cursor.close()

    result = {row["ticker"]: row["id"] for row in rows}

    missing = set(tickers) - set(result)

    if missing:
        raise RuntimeError(
            "Os seguintes ativos não existem na tabela assets: "
            + ", ".join(sorted(missing))
        )

    return result


# ============================================================
# YFINANCE COLLECTOR
# ============================================================

class YFinanceCollector:
    def __init__(self, settings: Settings):
        self.settings = settings

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza DataFrames retornados pelo yfinance.

        Dependendo da versão/configuração do yfinance,
        as colunas podem vir como MultiIndex.
        """

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                str(column[0]).lower()
                for column in df.columns
            ]
        else:
            df.columns = [
                str(column).lower()
                for column in df.columns
            ]

        return df

    def collect_asset(self, ticker: str) -> pd.DataFrame:
        logger.info(
            "Coletando dados de mercado: %s",
            ticker,
        )

        try:
            data = yf.download(
                ticker,
                period=self.settings.period,
                interval=self.settings.interval,
                auto_adjust=False,
                progress=False,
                timeout=self.settings.request_timeout,
                threads=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Falha ao consultar yfinance para {ticker}: {exc}"
            ) from exc

        if data is None or data.empty:
            raise RuntimeError(
                f"yfinance não retornou dados para {ticker}."
            )

        data = self._normalize_columns(data)

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        missing = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        if missing:
            raise RuntimeError(
                f"Colunas ausentes para {ticker}: {missing}"
            )

        # adjusted_close pode não estar presente em algumas versões.
        if "adj close" in data.columns:
            data["adjusted_close"] = data["adj close"]
        elif "adjusted_close" not in data.columns:
            data["adjusted_close"] = data["close"]

        data = data.reset_index()
        data = self._normalize_columns(data)

        date_column = "date" if "date" in data.columns else "datetime"

        if date_column not in data.columns:
            raise RuntimeError(
                f"Não foi possível identificar a coluna de data para {ticker}."
            )

        data["price_date"] = pd.to_datetime(
            data[date_column],
            errors="coerce",
        ).dt.date

        result = data[
            [
                "price_date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
            ]
        ].copy()

        result["ticker"] = ticker

        # Conversão numérica.
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
        ]

        for column in numeric_columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

        # Remove linhas inválidas.
        result = result.dropna(
            subset=[
                "price_date",
                "open",
                "high",
                "low",
                "close",
            ]
        )

        # Remove preços impossíveis.
        result = result[
            (result["open"] > 0)
            & (result["high"] > 0)
            & (result["low"] > 0)
            & (result["close"] > 0)
        ]

        # Volume pode vir como float do yfinance.
        result["volume"] = (
            result["volume"]
            .fillna(0)
            .clip(lower=0)
            .astype("int64")
        )

        result = result.drop_duplicates(
            subset=["price_date"]
        )

        result = result.sort_values(
            "price_date"
        ).reset_index(drop=True)

        logger.info(
            "%s: %s registros coletados.",
            ticker,
            len(result),
        )

        return result


# ============================================================
# MARKET PRICE VALIDATION
# ============================================================

def validate_market_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df

    invalid_ohlc = (
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    )

    invalid_rows = int(invalid_ohlc.sum())

    if invalid_rows:
        logger.warning(
            "Removendo %s registros com OHLC inconsistente.",
            invalid_rows,
        )

        df = df.loc[~invalid_ohlc].copy()

    return df


# ============================================================
# LOAD MARKET PRICES
# ============================================================

def load_market_prices(
    db: Database,
    asset_map: dict[str, int],
    collected: dict[str, pd.DataFrame],
) -> int:
    query = """
        INSERT INTO market_prices (
            asset_id,
            price_date,
            open_price,
            high_price,
            low_price,
            close_price,
            adjusted_close,
            volume,
            source
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            open_price = VALUES(open_price),
            high_price = VALUES(high_price),
            low_price = VALUES(low_price),
            close_price = VALUES(close_price),
            adjusted_close = VALUES(adjusted_close),
            volume = VALUES(volume),
            source = VALUES(source)
    """

    rows: list[tuple] = []

    for ticker, df in collected.items():
        asset_id = asset_map[ticker]

        for row in df.itertuples(index=False):
            rows.append(
                (
                    asset_id,
                    row.price_date,
                    float(row.open),
                    float(row.high),
                    float(row.low),
                    float(row.close),
                    float(row.adjusted_close),
                    int(row.volume),
                    "yfinance",
                )
            )

    affected = db.executemany(query, rows)

    logger.info(
        "Carga market_prices concluída. Registros afetados: %s",
        affected,
    )

    return affected


# ============================================================
# ETL - READ MARKET DATA
# ============================================================

def extract_market_data(
    db: Database,
    asset_id: int,
) -> pd.DataFrame:
    query = """
        SELECT
            price_date,
            close_price,
            volume
        FROM market_prices
        WHERE asset_id = %s
        ORDER BY price_date
    """

    cursor = db.cursor()

    try:
        cursor.execute(query, (asset_id,))
        rows = cursor.fetchall()
    finally:
        cursor.close()

    if not rows:
        return pd.DataFrame(
            columns=[
                "price_date",
                "close_price",
                "volume",
            ]
        )

    df = pd.DataFrame(rows)

    df["price_date"] = pd.to_datetime(
        df["price_date"]
    )

    df["close_price"] = pd.to_numeric(
        df["close_price"],
        errors="coerce",
    )

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce",
    )

    return df


# ============================================================
# ETL - FEATURE ENGINEERING
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria as features definidas para o MVP.

    IMPORTANTE:
        target_next_close é o fechamento do próximo pregão.
        Nenhum dado futuro é utilizado para construir as features.
    """

    if df.empty:
        return df

    result = df.copy()

    result = result.sort_values(
        "price_date"
    ).reset_index(drop=True)

    close = result["close_price"]

    # Retornos.
    result["return_1d"] = close.pct_change(1)
    result["return_5d"] = close.pct_change(5)
    result["return_20d"] = close.pct_change(20)

    # Médias móveis.
    result["sma_5"] = close.rolling(
        window=5,
        min_periods=5,
    ).mean()

    result["sma_10"] = close.rolling(
        window=10,
        min_periods=10,
    ).mean()

    result["sma_20"] = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    # Volatilidade dos retornos.
    daily_return = result["return_1d"]

    result["volatility_5d"] = daily_return.rolling(
        window=5,
        min_periods=5,
    ).std()

    result["volatility_20d"] = daily_return.rolling(
        window=20,
        min_periods=20,
    ).std()

    # Relação preço / média.
    result["price_sma5_ratio"] = (
        close / result["sma_5"]
    )

    result["price_sma20_ratio"] = (
        close / result["sma_20"]
    )

    # Target:
    # preço de fechamento do próximo pregão.
    #
    # Esta coluna é criada para deixar o dataset pronto
    # para a futura etapa de Machine Learning.
    #
    # Nenhum treinamento ou previsão é realizado aqui.
    result["target_next_close"] = close.shift(-1)

    return result


# ============================================================
# ETL - LOAD ANALYTICAL DATA
# ============================================================

def load_analytical_data(
    db: Database,
    asset_id: int,
    df: pd.DataFrame,
) -> int:
    if df.empty:
        return 0

    query = """
        INSERT INTO analytical_data (
            asset_id,
            price_date,
            close_price,
            volume,
            return_1d,
            return_5d,
            return_20d,
            sma_5,
            sma_10,
            sma_20,
            volatility_5d,
            volatility_20d,
            price_sma5_ratio,
            price_sma20_ratio,
            target_next_close
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            close_price = VALUES(close_price),
            volume = VALUES(volume),
            return_1d = VALUES(return_1d),
            return_5d = VALUES(return_5d),
            return_20d = VALUES(return_20d),
            sma_5 = VALUES(sma_5),
            sma_10 = VALUES(sma_10),
            sma_20 = VALUES(sma_20),
            volatility_5d = VALUES(volatility_5d),
            volatility_20d = VALUES(volatility_20d),
            price_sma5_ratio = VALUES(price_sma5_ratio),
            price_sma20_ratio = VALUES(price_sma20_ratio),
            target_next_close = VALUES(target_next_close)
    """

    rows: list[tuple] = []

    for row in df.itertuples(index=False):
        def value(field: str) -> Any:
            current = getattr(row, field)

            if pd.isna(current):
                return None

            if hasattr(current, "item"):
                return current.item()

            return current

        rows.append(
            (
                asset_id,
                row.price_date.date()
                if hasattr(row.price_date, "date")
                else row.price_date,

                value("close_price"),
                value("volume"),

                value("return_1d"),
                value("return_5d"),
                value("return_20d"),

                value("sma_5"),
                value("sma_10"),
                value("sma_20"),

                value("volatility_5d"),
                value("volatility_20d"),

                value("price_sma5_ratio"),
                value("price_sma20_ratio"),

                value("target_next_close"),
            )
        )

    affected = db.executemany(
        query,
        rows,
    )

    logger.info(
        "ETL analytical_data concluído para asset_id=%s. "
        "Registros afetados: %s",
        asset_id,
        affected,
    )

    return affected


# ============================================================
# ETL PIPELINE
# ============================================================

def run_etl(
    db: Database,
    asset_map: dict[str, int],
) -> int:
    total_processed = 0

    for ticker, asset_id in asset_map.items():
        logger.info(
            "Executando ETL para %s.",
            ticker,
        )

        raw_data = extract_market_data(
            db,
            asset_id,
        )

        if raw_data.empty:
            logger.warning(
                "Nenhum dado encontrado para ETL: %s",
                ticker,
            )
            continue

        analytical = build_features(
            raw_data
        )

        processed = load_analytical_data(
            db,
            asset_id,
            analytical,
        )

        total_processed += processed

    logger.info(
        "ETL finalizado. Total processado: %s",
        total_processed,
    )

    return total_processed


# ============================================================
# DATABASE HEALTH CHECK
# ============================================================

def database_health_check(db: Database) -> None:
    cursor = db.cursor()

    try:
        cursor.execute(
            "SELECT VERSION() AS version"
        )

        row = cursor.fetchone()

        logger.info(
            "MySQL conectado. Versão: %s",
            row["version"],
        )
    finally:
        cursor.close()


# ============================================================
# COLLECTION PIPELINE
# ============================================================

def run_collection(
    db: Database,
    settings: Settings,
    asset_map: dict[str, int],
) -> int:
    collector = YFinanceCollector(settings)

    collected: dict[str, pd.DataFrame] = {}

    for ticker in settings.assets:
        try:
            df = collector.collect_asset(
                ticker
            )

            df = validate_market_data(
                df
            )

            if not df.empty:
                collected[ticker] = df

        except Exception as exc:
            logger.exception(
                "Erro coletando %s: %s",
                ticker,
                exc,
            )

    if not collected:
        raise RuntimeError(
            "Nenhum ativo foi coletado com sucesso."
        )

    return load_market_prices(
        db,
        asset_map,
        collected,
    )


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_pipeline(
    settings: Settings,
) -> None:
    db = Database(settings)
    pipeline_run = None

    total_collected = 0
    total_processed = 0

    try:
        db.connect()
        database_health_check(db)

        asset_map = get_asset_map(
            db,
            settings.assets,
        )

        pipeline_run = PipelineRun(db)
        pipeline_run.start()

        # ----------------------------------------------------
        # 1. COLLECTION
        # ----------------------------------------------------

        logger.info("=== ETAPA 1/2: COLETA ===")

        total_collected = run_collection(
            db,
            settings,
            asset_map,
        )

        # ----------------------------------------------------
        # 2. ETL
        # ----------------------------------------------------

        logger.info("=== ETAPA 2/2: ETL ===")

        total_processed = run_etl(
            db,
            asset_map,
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Random Forest NÃO entra nesta versão.
        # Nenhum modelo é treinado.
        # Nenhuma previsão é gerada.
        # ----------------------------------------------------

        pipeline_run.finish(
            status="SUCCESS",
            records_collected=total_collected,
            records_processed=total_processed,
            predictions_generated=0,
        )

        logger.info(
            "Pipeline concluído com sucesso."
        )

    except Exception as exc:
        logger.exception(
            "Falha no pipeline: %s",
            exc,
        )

        if pipeline_run:
            try:
                pipeline_run.finish(
                    status="FAILED",
                    records_collected=total_collected,
                    records_processed=total_processed,
                    predictions_generated=0,
                    error_message=str(exc),
                )
            except Exception:
                logger.exception(
                    "Falha ao registrar o erro do pipeline."
                )

        raise

    finally:
        db.close()


# ============================================================
# COLLECTION ONLY
# ============================================================

def run_collect_only(
    settings: Settings,
) -> None:
    db = Database(settings)

    try:
        db.connect()
        database_health_check(db)

        asset_map = get_asset_map(
            db,
            settings.assets,
        )

        run_collection(
            db,
            settings,
            asset_map,
        )

        logger.info(
            "Coleta concluída."
        )

    finally:
        db.close()


# ============================================================
# ETL ONLY
# ============================================================

def run_etl_only(
    settings: Settings,
) -> None:
    db = Database(settings)

    try:
        db.connect()
        database_health_check(db)

        asset_map = get_asset_map(
            db,
            settings.assets,
        )

        run_etl(
            db,
            asset_map,
        )

        logger.info(
            "ETL concluído."
        )

    finally:
        db.close()


# ============================================================
# CLI
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline financeiro da Heineken "
            "sem Machine Learning."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "collect",
            "etl",
        ],
        default="full",
        help=(
            "full = coleta + ETL; "
            "collect = somente coleta; "
            "etl = somente ETL."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        settings = load_settings()

        logger.info(
            "=============================================="
        )
        logger.info(
            "HEINEKEN ANALYTICS - PIPELINE"
        )
        logger.info(
            "Modo: execução sem Machine Learning"
        )
        logger.info(
            "Ativos: %s",
            ", ".join(settings.assets),
        )
        logger.info(
            "=============================================="
        )

        args = parse_arguments()

        if args.mode == "full":
            run_full_pipeline(
                settings
            )

        elif args.mode == "collect":
            run_collect_only(
                settings
            )

        elif args.mode == "etl":
            run_etl_only(
                settings
            )

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Execução interrompida pelo usuário."
        )
        return 130

    except Exception as exc:
        logger.exception(
            "Execução encerrada com erro: %s",
            exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
