# Heineken Predictive Analytics 🍺

Um ecossistema modular em Python e MySQL para **engenharia de dados financeiros**, **análise exploratória de dados (EDA)**, **modelagem preditiva (Random Forest)** e **visualização em tempo real (Streamlit)** aplicado aos ativos do Grupo Heineken (`HEIA.AS` e `HEINY`).

---

## 📌 Visão Geral da Arquitetura

O pipeline é divido em 4 módulos principais que se comunicam através do banco relacional MySQL:

```text
[CRON / Manual] 
       │
       ▼
1. main.py (Coleta yfinance ➔ Validação OHLC ➔ Carga MySQL ➔ ETL/Features ➔ analytical_data)
       │
       ├───────────────────────────────┐
       ▼                               ▼
2. analysis.py (EDA / Relatórios)   3. model_training.py (ML Random Forest ➔ predictions)
                                       │
                                       ▼
                            4. app.py (Dashboard Streamlit)
```

---

## 🛠️ Tecnologias e Dependências

- **Linguagem:** Python 3.11+
- **Banco de Dados:** MySQL 8+
- **Coleta de Dados:** `yfinance`
- **Engenharia de Dados & ETL:** `pandas`, `numpy`, `mysql-connector-python`
- **Análise Exploratória:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `joblib`
- **Dashboard Interativo:** `streamlit`, `plotly`

---

## 🚀 Instalação e Configuração

### 1. Clonar o Repositório e Configurar o Ambiente Virtual

```bash
git clone https://github.com/seu-usuario/heineken-analytics.git
cd heineken-analytics
```

Crie e ative um ambiente virtual:

- **Windows:**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  ```
- **Linux/macOS:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

Instale as dependências exigidas pelo projeto:

```bash
pip install -r requirements.txt
```

---

### 2. Configuração do Banco de Dados MySQL

1. Execute o script `schema.sql` no seu servidor MySQL para criar a base de dados e suas tabelas estruturadas:
   ```bash
   mysql -u root -p < schema.sql
   ```
2. Duplique o arquivo `.env.example` para `.env`:
   - Windows: `copy .env.example .env`
   - Linux/macOS: `cp .env.example .env`
3. Abra o arquivo `.env` e atualize as credenciais do seu banco local/remoto:
   ```ini
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_DATABASE=heineken_analytics
   MYSQL_USER=root
   MYSQL_PASSWORD=sua_senha_aqui

   ASSETS=HEIA.AS,HEINY
   YFINANCE_PERIOD=5y
   YFINANCE_INTERVAL=1d
   YFINANCE_TIMEOUT=30
   ```

---

## ⚙️ Modos de Execução

### 1. Pipeline de Dados (Coleta & ETL) — `main.py`

Executa a ingestão idempotente no banco `market_prices` e constrói a tabela `analytical_data` com indicadores técnicos (retornos de 1d, 5d e 20d, SMAs de 5, 10 e 20 dias, e volatilidades móveis).

- **Pipeline Completo (Coleta + ETL):**
  ```bash
  python main.py --mode full
  ```
- **Somente Coleta de Dados:**
  ```bash
  python main.py --mode collect
  ```
- **Somente Processamento ETL:**
  ```bash
  python main.py --mode etl
  ```

---

### 2. Análise Exploratória de Dados (EDA) — `analysis.py`

Lê a tabela `analytical_data`, calcula estatísticas descritivas no console e exporta relatórios visuais de alta resolução na pasta `reports/figures/`:

```bash
python analysis.py
```

*Parâmetros opcionais:*
```bash
python analysis.py --tickers HEIA.AS --dpi 200 --output-dir reports/figures
```

---

### 3. Treinamento de Machine Learning — `model_training.py`

Treina um modelo `RandomForestRegressor` para prever o fechamento do próximo pregão (`target_next_close`). Utiliza divisão cronológica sem embaralhamento e validação cruzada temporal (`TimeSeriesSplit`) para prevenir vazamento de dados (*data leakage*).

- Treina o modelo, avalia no teste (MAE, RMSE, R² vs. Baseline "amanhã = hoje"), salva o artefato `.joblib` em `models/` e persiste as previsões na tabela `predictions`:
  ```bash
  python model_training.py
  ```

*Parâmetros opcionais:*
```bash
python model_training.py --model-version 1.1.0 --n-estimators 500 --refit-full
```

---

### 4. Dashboard Interativo — `app.py`

Inicia a aplicação web interativa em Streamlit para visualização dos KPIs de mercado, gráficos Plotly de preços, volatilidade, matrizes e comparação do modelo предittivo contra o valor real.

```bash
streamlit run app.py
```

Acesse a interface no seu navegador através do endereço `http://localhost:8501`.

---

## ⏰ Automação Diária (CRON)

Para manter os dados de mercado e as tabelas analíticas atualizados de forma automática todos os dias úteis após o fechamento do pregão (ex: 18:30h), você pode adicionar uma tarefa no CRON do servidor Linux:

```cron
30 18 * * 1-5 cd /caminho/do/projeto && /caminho/do/projeto/.venv/bin/python main.py --mode full >> logs/cron.log 2>&1
```

---

## 📂 Estrutura do Projeto

```text
.
├── data/                    # Dados auxiliares / fallback estático
├── logs/                    # Arquivos de log de execução
├── models/                  # Modelos Random Forest treinados (.joblib)
├── reports/
│   └── figures/             # Gráficos gerados pelo módulo de EDA (analysis.py)
├── .env.example             # Modelo de variáveis de ambiente
├── analysis.py              # Módulo de Análise Exploratória (EDA)
├── app.py                   # Dashboard interativo (Streamlit)
├── main.py                  # Pipeline principal de dados (yfinance -> MySQL)
├── model_training.py        # Módulo de treino e previsão (Random Forest)
├── README.md                # Documentação completa do projeto
├── requirements.txt         # Lista de dependências Python
└── schema.sql               # Estrutura do banco de dados MySQL
```

---

## 🛡️ Licença e Aviso Legal

Este projeto tem fins estritamente acadêmicos e educacionais. As previsões geradas pelo modelo estatístico não constituem qualquer tipo de recomendação financeira ou de investimento.