from pathlib import Path
import pandas as pd
from main import Database, load_settings

def export_all_data() -> None:
    settings = load_settings()
    db = Database(settings)
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        db.connect()
        
        # 1. Exporta analytical_data
        print("Exportando analytical_data...")
        query_analytical = """
            SELECT 
                a.ticker,
                ad.price_date,
                ad.close_price,
                ad.volume,
                ad.return_1d,
                ad.return_5d,
                ad.return_20d,
                ad.sma_5,
                ad.sma_10,
                ad.sma_20,
                ad.volatility_5d,
                ad.volatility_20d,
                ad.price_sma5_ratio,
                ad.price_sma20_ratio,
                ad.target_next_close
            FROM analytical_data AS ad
            INNER JOIN assets AS a ON a.id = ad.asset_id
            ORDER BY a.ticker, ad.price_date
        """
        cursor = db.cursor()
        cursor.execute(query_analytical)
        rows_analytical = cursor.fetchall()
        
        if rows_analytical:
            df_analytical = pd.DataFrame(rows_analytical)
            df_analytical.to_csv(data_dir / "analytical_data.csv", index=False, encoding="utf-8")
            print(" -> data/analytical_data.csv gerado com sucesso!")
        
        # 2. Exporta predictions (Machine Learning)
        print("Exportando predictions...")
        query_predictions = """
            SELECT 
                a.ticker,
                p.prediction_date,
                p.target_date,
                p.actual_value,
                p.predicted_value,
                p.model_name,
                p.model_version,
                p.mae,
                p.rmse,
                p.r2,
                p.created_at
            FROM predictions AS p
            INNER JOIN assets AS a ON a.id = p.asset_id
            ORDER BY a.ticker, p.prediction_date
        """
        cursor.execute(query_predictions)
        rows_predictions = cursor.fetchall()
        cursor.close()
        
        if rows_predictions:
            df_predictions = pd.DataFrame(rows_predictions)
            df_predictions.to_csv(data_dir / "predictions.csv", index=False, encoding="utf-8")
            print(" -> data/predictions.csv gerado com sucesso!")
        else:
            print(" AVISO: Nenhuma previsão encontrada na tabela `predictions`. Execute `python model_training.py` antes de exportar.")

    finally:
        db.close()

if __name__ == "__main__":
    export_all_data()