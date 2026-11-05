from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from mysql.connector import Error as MySQLError
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit

from main import (
    Database,
    PipelineRun,
    Settings,
    database_health_check,
    get_asset_map,
    load_settings,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CAMINHOS, CONSTANTES E COLUNAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = BASE_DIR / "models"

MODEL_NAME = "random_forest_regressor"
DEFAULT_MODEL_VERSION = "1.0.0"

TARGET_COLUMN = "target_next_close"

# Features geradas pelo ETL (build_features em main.py).
FEATURE_COLUMNS: tuple[str, ...] = (
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
)

MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 20


# ============================================================
# LOGGING (LOGGER)
# ============================================================

def _build_logger() -> logging.Logger:
    """
    Logger do módulo, com arquivo próprio (logs/training.log) e
    console, no mesmo formato do pipeline principal.

    `propagate=False` evita linhas duplicadas, já que o main.py
    configura o logger raiz ao ser importado.
    """

    log = logging.getLogger("heineken_training")

    if log.handlers:
        return log

    log.setLevel(logging.INFO)
    log.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "training.log",
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
# EXCEÇÕES E TIPOS DE DADOS
# ============================================================

class TrainingError(RuntimeError):
    """Erro de negócio/dados na etapa de treinamento."""


@dataclass(frozen=True)
class TrainingConfig:
    """Hiperparâmetros e opções de execução do treinamento."""

    test_size: float = 0.2
    cv_splits: int = 5
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 3
    random_state: int = 42
    model_version: str = DEFAULT_MODEL_VERSION
    models_dir: Path = MODELS_DIR
    refit_full: bool = False


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class AssetResult:
    ticker: str
    asset_id: int
    train_rows: int
    test_rows: int
    metrics: Metrics
    baseline: Metrics
    predictions_saved: int
    model_path: Path


@dataclass(frozen=True)
class TrainingSummary:
    results: list[AssetResult]
    failures: dict[str, str]


def validate_config(config: TrainingConfig) -> None:
    """Valida os parâmetros antes de acessar o banco."""

    if not 0.05 <= config.test_size <= 0.5:
        raise TrainingError("test_size deve estar entre 0.05 e 0.5.")

    if config.cv_splits < 2:
        raise TrainingError("cv_splits deve ser pelo menos 2.")

    if config.n_estimators < 1:
        raise TrainingError("n_estimators deve ser pelo menos 1.")

    if config.min_samples_leaf < 1:
        raise TrainingError("min_samples_leaf deve ser pelo menos 1.")

    if not config.model_version.strip():
        raise TrainingError("model_version não pode ser vazio.")


# ============================================================
# ETAPA 1 - LEITURA DE DADOS
# ============================================================

def load_asset_dataset(
    db: Database,
    ticker: str,
    asset_id: int,
) -> pd.DataFrame:
    """Lê `analytical_data` de UM ativo, ordenado por data."""

    columns = ", ".join(FEATURE_COLUMNS + (TARGET_COLUMN,))

    # `columns` vem de constantes do módulo; o asset_id é parametrizado.
    query = f"""
        SELECT
            price_date,
            {columns}
        FROM analytical_data
        WHERE asset_id = %s
        ORDER BY price_date
    """

    cursor = db.cursor()

    try:
        cursor.execute(query, (asset_id,))
        rows = cursor.fetchall()
    except MySQLError as exc:
        raise TrainingError(
            f"Falha ao ler analytical_data de {ticker}: {exc}"
        ) from exc
    finally:
        cursor.close()

    if not rows:
        raise TrainingError(
            f"Sem dados em analytical_data para {ticker}. "
            "Execute antes o ETL (python main.py --mode etl)."
        )

    df = pd.DataFrame(rows)

    df["price_date"] = pd.to_datetime(df["price_date"])

    for column in FEATURE_COLUMNS + (TARGET_COLUMN,):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    logger.info("%s: %s linhas lidas de analytical_data.", ticker, len(df))

    return df


# ============================================================
# ETAPA 2 - PREPARAÇÃO DE DADOS E SPLIT CRONOLÓGICO
# ============================================================

def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa o dataset em:

    - `supervised`: linhas com todas as features E o target
      (usadas para treino e teste).
    - `latest`: a linha mais recente, que tem features mas ainda
      não tem target (o "amanhã" ainda não aconteceu). Serve para a
      previsão do próximo pregão.

    Também cria `target_date`: a data do fechamento que o target
    representa (a data da linha seguinte).
    """

    data = df.sort_values("price_date").reset_index(drop=True)

    data["target_date"] = data["price_date"].shift(-1)

    features_ready = data[list(FEATURE_COLUMNS)].notna().all(axis=1)
    has_target = data[TARGET_COLUMN].notna()

    supervised = data[features_ready & has_target].copy()

    latest = data[
        features_ready
        & ~has_target
        & (data["price_date"] == data["price_date"].max())
    ].copy()

    if not latest.empty:
        # Próximo dia útil (não considera feriados da bolsa).
        latest["target_date"] = latest["price_date"] + pd.offsets.BDay(1)

    return supervised.reset_index(drop=True), latest.reset_index(drop=True)


def chronological_split(
    data: pd.DataFrame,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corte cronológico: os primeiros (1 - test_size) dos dados são
    treino e o restante é teste. Nunca há embaralhamento.
    """

    split_index = int(len(data) * (1 - test_size))

    train = data.iloc[:split_index].copy()
    test = data.iloc[split_index:].copy()

    if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
        raise TrainingError(
            f"Dados insuficientes para o split: treino={len(train)} "
            f"(mín. {MIN_TRAIN_ROWS}), teste={len(test)} "
            f"(mín. {MIN_TEST_ROWS})."
        )

    # Trava de segurança contra vazamento temporal.
    if train["price_date"].max() >= test["price_date"].min():
        raise TrainingError(
            "Split inválido: o treino não é estritamente anterior ao teste."
        )

    return train, test


# ============================================================
# ETAPA 3 - TREINAMENTO DO MODELO
# ============================================================

def build_model(config: TrainingConfig) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
        n_jobs=-1,
    )


def cross_validate_time_series(
    model: RandomForestRegressor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
) -> list[float]:
    """
    Validação cruzada temporal (TimeSeriesSplit) APENAS no treino.

    Em cada fold, o modelo treina em um bloco inicial e valida no
    bloco imediatamente posterior. O conjunto de teste final nunca
    é usado aqui.
    """

    if len(x_train) < (n_splits + 1) * 20:
        logger.warning(
            "Treino pequeno (%s linhas): validação cruzada ignorada.",
            len(x_train),
        )
        return []

    fold_maes: list[float] = []

    try:
        splitter = TimeSeriesSplit(n_splits=n_splits)

        for fold, (fit_idx, val_idx) in enumerate(
            splitter.split(x_train), start=1
        ):
            fold_model = clone(model)
            fold_model.fit(x_train.iloc[fit_idx], y_train.iloc[fit_idx])

            predictions = fold_model.predict(x_train.iloc[val_idx])
            mae = float(
                mean_absolute_error(y_train.iloc[val_idx], predictions)
            )

            fold_maes.append(mae)

            logger.info(
                "  CV fold %s/%s: treino=%s validação=%s MAE=%.4f",
                fold,
                n_splits,
                len(fit_idx),
                len(val_idx),
                mae,
            )
    except ValueError as exc:
        logger.warning("Validação cruzada ignorada: %s", exc)
        return []

    return fold_maes


def fit_model(
    model: RandomForestRegressor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    try:
        model.fit(x_train, y_train)
    except ValueError as exc:
        raise TrainingError(f"Falha ao treinar o modelo: {exc}") from exc

    return model


# ============================================================
# ETAPA 4 - AVALIAÇÃO DO MODELO
# ============================================================

def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> Metrics:
    """Calcula MAE, RMSE e R²."""

    return Metrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
    )


def log_metrics(
    ticker: str,
    model_metrics: Metrics,
    baseline: Metrics,
    fold_maes: list[float],
    model: RandomForestRegressor,
) -> None:
    logger.info("-" * 60)
    logger.info("RESULTADOS - %s (período de teste)", ticker)
    logger.info("-" * 60)
    logger.info(
        "%-22s MAE=%.4f  RMSE=%.4f  R²=%.4f",
        "Random Forest",
        model_metrics.mae,
        model_metrics.rmse,
        model_metrics.r2,
    )
    logger.info(
        "%-22s MAE=%.4f  RMSE=%.4f  R²=%.4f",
        "Baseline (amanhã=hoje)",
        baseline.mae,
        baseline.rmse,
        baseline.r2,
    )

    if fold_maes:
        logger.info(
            "CV temporal (treino): MAE médio=%.4f ± %.4f",
            float(np.mean(fold_maes)),
            float(np.std(fold_maes)),
        )

    if model_metrics.mae < baseline.mae:
        logger.info("O modelo SUPEROU o baseline em MAE.")
    else:
        logger.warning(
            "O modelo NÃO superou o baseline em MAE. "
            "Interprete as previsões com cautela."
        )

    importances = pd.Series(
        model.feature_importances_,
        index=list(FEATURE_COLUMNS),
    ).sort_values(ascending=False)

    logger.info(
        "Top 5 features: %s",
        ", ".join(
            f"{name}={value:.3f}"
            for name, value in importances.head(5).items()
        ),
    )


# ============================================================
# ETAPA 5 - SALVAR MODELO (JOBLIB)
# ============================================================

def _slug(text: str) -> str:
    """Converte um ticker em nome de arquivo seguro (HEIA.AS -> HEIA_AS)."""

    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def save_model(
    model: RandomForestRegressor,
    ticker: str,
    asset_id: int,
    config: TrainingConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    metrics: Metrics,
    baseline: Metrics,
) -> Path:
    """
    Salva o modelo com metadados em `models/`.

    A gravação é feita em arquivo temporário e depois renomeada,
    para nunca deixar um .joblib corrompido caso algo falhe no meio.
    """

    artifact: dict[str, Any] = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "ticker": ticker,
        "asset_id": asset_id,
        "model_name": MODEL_NAME,
        "model_version": config.model_version,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "trained_on": "train+test" if config.refit_full else "train",
        "train_period": (
            str(train["price_date"].min().date()),
            str(train["price_date"].max().date()),
        ),
        "test_period": (
            str(test["price_date"].min().date()),
            str(test["price_date"].max().date()),
        ),
        "metrics": asdict(metrics),
        "baseline_metrics": asdict(baseline),
        "hyperparameters": model.get_params(),
        "sklearn_version": sklearn.__version__,
    }

    path = (
        config.models_dir
        / f"rf_{_slug(ticker)}_{_slug(config.model_version)}.joblib"
    )
    temp_path = path.with_suffix(".joblib.tmp")

    try:
        config.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, temp_path, compress=3)
        temp_path.replace(path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise TrainingError(
            f"Falha ao salvar o modelo em {path}: {exc}"
        ) from exc

    logger.info("Modelo salvo: %s", path)

    return path


def load_model_artifact(path: Path) -> dict[str, Any]:
    """
    Carrega um modelo salvo por `save_model` (uso futuro/inferência).

    Atenção: arquivos joblib/pickle só devem ser carregados de fontes
    confiáveis, pois podem executar código arbitrário.
    """

    try:
        artifact = joblib.load(path)
    except (OSError, EOFError) as exc:
        raise TrainingError(
            f"Falha ao carregar o modelo {path}: {exc}"
        ) from exc

    required = {"model", "feature_columns", "model_version"}

    if not isinstance(artifact, dict) or not required <= artifact.keys():
        raise TrainingError(f"Arquivo de modelo inválido: {path}")

    return artifact


# ============================================================
# ETAPA 6 - SALVAR PREVISÕES (TABELA predictions MYSQL)
# ============================================================

def _db_float(value: float | None) -> float | None:
    """Converte para float aceito pelo MySQL (NaN/inf viram NULL)."""

    if value is None:
        return None

    number = float(value)

    return number if math.isfinite(number) else None


def build_prediction_rows(
    asset_id: int,
    config: TrainingConfig,
    metrics: Metrics,
    test: pd.DataFrame,
    test_predictions: np.ndarray,
    latest: pd.DataFrame,
    latest_predictions: np.ndarray,
) -> list[tuple]:
    """
    Monta as linhas da tabela `predictions`.

    - prediction_date: data em que a informação (features) foi
      observada, ou seja, o dia em que a previsão poderia ser feita.
    - target_date: data do fechamento que está sendo previsto.
    - actual_value: preço real (NULL para o próximo pregão, ainda
      desconhecido).
    - mae/rmse/r2: métricas do período de teste do modelo, repetidas
      em cada linha (o schema guarda as métricas por linha).
    """

    mae = _db_float(metrics.mae)
    rmse = _db_float(metrics.rmse)
    r2 = _db_float(metrics.r2)

    rows: list[tuple] = []

    for row, predicted in zip(
        test.itertuples(index=False), test_predictions
    ):
        rows.append(
            (
                asset_id,
                row.price_date.date(),
                row.target_date.date(),
                _db_float(getattr(row, TARGET_COLUMN)),
                _db_float(predicted),
                MODEL_NAME,
                config.model_version,
                mae,
                rmse,
                r2,
            )
        )

    for row, predicted in zip(
        latest.itertuples(index=False), latest_predictions
    ):
        rows.append(
            (
                asset_id,
                row.price_date.date(),
                row.target_date.date(),
                None,
                _db_float(predicted),
                MODEL_NAME,
                config.model_version,
                mae,
                rmse,
                r2,
            )
        )

    return rows


def save_predictions(
    db: Database,
    asset_id: int,
    model_version: str,
    rows: list[tuple],
) -> int:
    """
    Grava as previsões de forma IDEMPOTENTE e ATÔMICA.

    A tabela `predictions` não tem chave única, então reexecutar o
    script duplicaria linhas. Para evitar isso, as previsões anteriores
    do mesmo (ativo, modelo, versão) são removidas e as novas inseridas
    na MESMA transação: se algo falhar, nada é alterado.
    """

    if not rows:
        return 0

    if db.connection is None:
        raise ConnectionError("Banco não conectado.")

    delete_query = """
        DELETE FROM predictions
        WHERE asset_id = %s
          AND model_name = %s
          AND model_version = %s
    """

    insert_query = """
        INSERT INTO predictions (
            asset_id,
            prediction_date,
            target_date,
            actual_value,
            predicted_value,
            model_name,
            model_version,
            mae,
            rmse,
            r2
        )
        VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """

    connection = db.connection
    cursor = connection.cursor()

    try:
        cursor.execute(
            delete_query,
            (asset_id, MODEL_NAME, model_version),
        )
        removed = cursor.rowcount

        cursor.executemany(insert_query, rows)
        inserted = cursor.rowcount

        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        cursor.close()

    logger.info(
        "predictions: %s removidas (execução anterior), %s inseridas "
        "(asset_id=%s, versão=%s).",
        removed,
        inserted,
        asset_id,
        model_version,
    )

    return int(inserted)


# ============================================================
# ETAPA 7 - TREINAMENTO DE UM ÚNICO ATIVO
# ============================================================

def train_asset(
    db: Database,
    ticker: str,
    asset_id: int,
    config: TrainingConfig,
) -> AssetResult:
    """Executa todo o fluxo de ML para um único ativo."""

    logger.info("=" * 60)
    logger.info("TREINAMENTO - %s (asset_id=%s)", ticker, asset_id)
    logger.info("=" * 60)

    # 1. Dados
    raw = load_asset_dataset(db, ticker, asset_id)
    supervised, latest = prepare_dataset(raw)

    # 2. Split cronológico
    train, test = chronological_split(supervised, config.test_size)

    logger.info(
        "%s: treino=%s linhas (%s a %s) | teste=%s linhas (%s a %s)",
        ticker,
        len(train),
        train["price_date"].min().date(),
        train["price_date"].max().date(),
        len(test),
        test["price_date"].min().date(),
        test["price_date"].max().date(),
    )

    features = list(FEATURE_COLUMNS)

    x_train, y_train = train[features], train[TARGET_COLUMN]
    x_test, y_test = test[features], test[TARGET_COLUMN]

    # 3. Validação cruzada temporal (somente no treino)
    base_model = build_model(config)

    fold_maes = cross_validate_time_series(
        base_model,
        x_train,
        y_train,
        config.cv_splits,
    )

    # 4. Treino final
    model = fit_model(clone(base_model), x_train, y_train)

    # 5. Avaliação no teste + baseline ingênuo
    test_predictions = model.predict(x_test)

    metrics = evaluate(y_test, test_predictions)
    baseline = evaluate(y_test, test["close_price"])

    log_metrics(ticker, metrics, baseline, fold_maes, model)

    # Modelo de produção: opcionalmente reajustado com TODOS os dados.
    # As métricas continuam sendo as do modelo avaliado no teste.
    production_model = model

    if config.refit_full:
        logger.info(
            "%s: reajustando o modelo com treino+teste (--refit-full).",
            ticker,
        )
        production_model = fit_model(
            clone(base_model),
            supervised[features],
            supervised[TARGET_COLUMN],
        )

    # Previsão do próximo pregão (sem valor real ainda).
    if latest.empty:
        latest_predictions = np.array([])
        logger.warning(
            "%s: sem linha recente com features completas; "
            "previsão do próximo pregão não gerada.",
            ticker,
        )
    else:
        latest_predictions = production_model.predict(latest[features])

        logger.info(
            "%s: previsão para %s = %.4f (último fechamento: %.4f)",
            ticker,
            latest["target_date"].iloc[0].date(),
            float(latest_predictions[0]),
            float(latest["close_price"].iloc[0]),
        )

    # 6. Salvar modelo
    model_path = save_model(
        production_model,
        ticker,
        asset_id,
        config,
        train,
        test,
        metrics,
        baseline,
    )

    # 7. Salvar previsões
    rows = build_prediction_rows(
        asset_id,
        config,
        metrics,
        test,
        test_predictions,
        latest,
        latest_predictions,
    )

    saved = save_predictions(db, asset_id, config.model_version, rows)

    return AssetResult(
        ticker=ticker,
        asset_id=asset_id,
        train_rows=len(train),
        test_rows=len(test),
        metrics=metrics,
        baseline=baseline,
        predictions_saved=saved,
        model_path=model_path,
    )


# ============================================================
# ORQUESTRAÇÃO DE TREINAMENTO DE VÁRIOS ATIVOS
# ============================================================

def log_summary(summary: TrainingSummary) -> None:
    logger.info("=" * 60)
    logger.info("RESUMO DO TREINAMENTO")
    logger.info("=" * 60)

    for result in summary.results:
        logger.info(
            "%-8s MAE=%.4f RMSE=%.4f R²=%.4f | baseline MAE=%.4f | "
            "previsões gravadas=%s",
            result.ticker,
            result.metrics.mae,
            result.metrics.rmse,
            result.metrics.r2,
            result.baseline.mae,
            result.predictions_saved,
        )

    for ticker, message in summary.failures.items():
        logger.error("%-8s FALHOU: %s", ticker, message)


def run_training(
    settings: Settings,
    tickers: tuple[str, ...] | None = None,
    config: TrainingConfig | None = None,
) -> TrainingSummary:
    """
    Treina um modelo por ativo e registra a execução em `pipeline_runs`.

    Um ativo que falha não impede os demais. Só levanta exceção se
    NENHUM ativo for treinado com sucesso.

    Pode ser chamada por outros módulos:

        from model_training import run_training
        run_training(load_settings())
    """

    config = config or TrainingConfig()
    validate_config(config)

    selected = tickers or settings.assets

    db = Database(settings)
    pipeline_run: PipelineRun | None = None

    results: list[AssetResult] = []
    failures: dict[str, str] = {}

    try:
        db.connect()
        database_health_check(db)

        asset_map = get_asset_map(db, selected)

        pipeline_run = PipelineRun(db)
        pipeline_run.start()

        for ticker, asset_id in asset_map.items():
            try:
                results.append(
                    train_asset(db, ticker, asset_id, config)
                )
            except Exception as exc:
                logger.exception(
                    "Falha no treinamento de %s: %s",
                    ticker,
                    exc,
                )
                failures[ticker] = str(exc)

        if not results:
            raise TrainingError(
                "Nenhum ativo foi treinado com sucesso: "
                + "; ".join(
                    f"{ticker}: {message}"
                    for ticker, message in failures.items()
                )
            )

        pipeline_run.finish(
            status="SUCCESS",
            records_processed=sum(
                r.train_rows + r.test_rows for r in results
            ),
            predictions_generated=sum(
                r.predictions_saved for r in results
            ),
            error_message=(
                "Ativos com falha: " + ", ".join(failures)
                if failures
                else None
            ),
        )

    except Exception as exc:
        if pipeline_run:
            try:
                pipeline_run.finish(
                    status="FAILED",
                    error_message=str(exc),
                )
            except Exception:
                logger.exception(
                    "Falha ao registrar o erro do treinamento."
                )

        raise

    finally:
        db.close()

    summary = TrainingSummary(results=results, failures=failures)
    log_summary(summary)

    return summary


# ============================================================
# INTERFACE DE LINHA DE COMANDO (CLI)
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Treina um Random Forest por ativo para prever "
            "o fechamento do próximo pregão."
        )
    )

    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Tickers a treinar (padrão: variável ASSETS do .env).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fração final da série usada como teste (padrão: 0.2).",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Nº de folds do TimeSeriesSplit no treino (padrão: 5).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Nº de árvores da floresta (padrão: 300).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidade máxima das árvores (padrão: sem limite).",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=3,
        help="Mínimo de amostras por folha (padrão: 3).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente de reprodutibilidade (padrão: 42).",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=(
            "Versão gravada em predictions e no nome do arquivo. "
            "Reexecutar com a mesma versão SUBSTITUI as previsões "
            f"anteriores (padrão: {DEFAULT_MODEL_VERSION})."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Pasta onde os modelos .joblib são salvos (padrão: models/).",
    )
    parser.add_argument(
        "--refit-full",
        action="store_true",
        help=(
            "Após avaliar, reajusta o modelo com treino+teste para a "
            "previsão do próximo pregão e para o arquivo salvo."
        ),
    )

    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        args = parse_arguments()
        settings = load_settings()

        config = TrainingConfig(
            test_size=args.test_size,
            cv_splits=args.cv_splits,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            model_version=args.model_version,
            models_dir=args.models_dir,
            refit_full=args.refit_full,
        )

        tickers = (
            tuple(ticker.strip().upper() for ticker in args.tickers)
            if args.tickers
            else None
        )

        logger.info("==============================================")
        logger.info("HEINEKEN ANALYTICS - TREINAMENTO RANDOM FOREST")
        logger.info(
            "Ativos: %s",
            ", ".join(tickers or settings.assets),
        )
        logger.info("Versão do modelo: %s", config.model_version)
        logger.info("==============================================")

        summary = run_training(settings, tickers, config)

        if summary.failures:
            logger.warning(
                "Treinamento concluído com falhas em: %s",
                ", ".join(summary.failures),
            )
            return 1

        logger.info("Treinamento concluído com sucesso.")
        return 0

    except KeyboardInterrupt:
        logger.warning("Execução interrompida pelo usuário.")
        return 130

    except (TrainingError, MySQLError, ConnectionError) as exc:
        logger.error("Falha no treinamento: %s", exc)
        return 1

    except Exception as exc:
        logger.exception("Execução encerrada com erro: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())