CREATE DATABASE IF NOT EXISTS heineken_analytics
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE heineken_analytics;

CREATE TABLE IF NOT EXISTS assets (
    id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    name VARCHAR(150) NOT NULL,
    exchange VARCHAR(100),
    currency CHAR(3),
    country VARCHAR(100),
    asset_type VARCHAR(50) NOT NULL DEFAULT 'stock',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT uq_assets_ticker UNIQUE (ticker)
);

CREATE TABLE IF NOT EXISTS market_prices (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    asset_id INT UNSIGNED NOT NULL,
    price_date DATE NOT NULL,
    open_price DECIMAL(15,4),
    high_price DECIMAL(15,4),
    low_price DECIMAL(15,4),
    close_price DECIMAL(15,4),
    adjusted_close DECIMAL(15,4),
    volume BIGINT UNSIGNED,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_market_prices_asset
        FOREIGN KEY (asset_id) REFERENCES assets(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT uq_market_prices_asset_date
        UNIQUE (asset_id, price_date),
    INDEX idx_market_prices_asset_date (asset_id, price_date),
    INDEX idx_market_prices_date (price_date)
);

CREATE TABLE IF NOT EXISTS financial_metrics (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    asset_id INT UNSIGNED NOT NULL,
    reference_date DATE NOT NULL,
    revenue DECIMAL(20,2),
    net_income DECIMAL(20,2),
    ebitda DECIMAL(20,2),
    total_assets DECIMAL(20,2),
    total_liabilities DECIMAL(20,2),
    cash DECIMAL(20,2),
    earnings_per_share DECIMAL(15,4),
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_financial_metrics_asset
        FOREIGN KEY (asset_id) REFERENCES assets(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT uq_financial_metrics_asset_date
        UNIQUE (asset_id, reference_date),
    INDEX idx_financial_metrics_asset_date (asset_id, reference_date)
);

CREATE TABLE IF NOT EXISTS analytical_data (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    asset_id INT UNSIGNED NOT NULL,
    price_date DATE NOT NULL,
    close_price DECIMAL(15,4),
    volume BIGINT UNSIGNED,
    return_1d DECIMAL(12,6),
    return_5d DECIMAL(12,6),
    return_20d DECIMAL(12,6),
    sma_5 DECIMAL(15,4),
    sma_10 DECIMAL(15,4),
    sma_20 DECIMAL(15,4),
    volatility_5d DECIMAL(12,6),
    volatility_20d DECIMAL(12,6),
    price_sma5_ratio DECIMAL(12,6),
    price_sma20_ratio DECIMAL(12,6),
    target_next_close DECIMAL(15,4),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_analytical_data_asset
        FOREIGN KEY (asset_id) REFERENCES assets(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    CONSTRAINT uq_analytical_data_asset_date
        UNIQUE (asset_id, price_date),
    INDEX idx_analytical_data_asset_date (asset_id, price_date)
);

CREATE TABLE IF NOT EXISTS predictions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    asset_id INT UNSIGNED NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    actual_value DECIMAL(15,4),
    predicted_value DECIMAL(15,4),
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    mae DECIMAL(15,6),
    rmse DECIMAL(15,6),
    r2 DECIMAL(15,6),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_predictions_asset
        FOREIGN KEY (asset_id) REFERENCES assets(id)
        ON UPDATE CASCADE ON DELETE RESTRICT,
    INDEX idx_predictions_asset_date (asset_id, prediction_date),
    INDEX idx_predictions_target_date (target_date)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    execution_date DATETIME NOT NULL,
    status ENUM('RUNNING', 'SUCCESS', 'FAILED') NOT NULL,
    records_collected INT UNSIGNED NOT NULL DEFAULT 0,
    records_processed INT UNSIGNED NOT NULL DEFAULT 0,
    predictions_generated INT UNSIGNED NOT NULL DEFAULT 0,
    error_message TEXT,
    started_at DATETIME NOT NULL,
    finished_at DATETIME,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pipeline_runs_execution_date (execution_date),
    INDEX idx_pipeline_runs_status (status)
);

INSERT INTO assets
    (ticker, name, exchange, currency, country, asset_type, active)
VALUES
    ('HEIA.AS', 'Heineken N.V.', 'Euronext Amsterdam', 'EUR',
     'Netherlands', 'stock', TRUE),
    ('HEINY', 'Heineken N.V. ADR', 'OTC Markets', 'USD',
     'Netherlands', 'ADR', TRUE)
ON DUPLICATE KEY UPDATE
    name = VALUES(name),
    exchange = VALUES(exchange),
    currency = VALUES(currency),
    country = VALUES(country),
    asset_type = VALUES(asset_type),
    active = VALUES(active);
