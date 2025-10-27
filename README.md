# 💰 Investo - AI-Powered Stock Analysis Platform

A comprehensive stock analysis application featuring:

## 🎯 Features
- **Investment Calculator**: Calculate expected returns with interactive charts
- **AI Stock Analysis**: OpenAI-powered stock insights and recommendations  
- **Dynamic Ticker Management**: Add new stocks with automated pipeline
- **Real-time Data**: Live stock data from Yahoo Finance
- **ML Predictions**: LSTM-based price predictions
- **Professional UI**: Modern, responsive design with beautiful tables

## 📊 Project Architecture & Data Flow

![Investo Project Flowchart: Data Ingestion, ML Pipeline, AI Analysis, and New Ticker Workflow](data/other/Screenshot%202025-10-26%20233829.png)

*This diagram shows the complete data flow from Yahoo Finance ingestion through ML pipeline to AI analysis, plus the automated ticker addition workflow.*

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run investment_calculator.py
```

### Deployment Options
This app is ready for deployment on:
- **Streamlit Cloud** (recommended - FREE)
- **Railway** (FREE tier)
- **Heroku** (FREE tier)
- **DigitalOcean App Platform**

## 📊 Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python, DuckDB
- **ML**: PyTorch, Scikit-learn
- **Data**: Yahoo Finance API
- **AI**: OpenAI GPT API
- **Visualization**: Plotly
- **Architecture**: Modular design with shared feature engineering

## 🔧 Configuration
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 📈 Usage
1. Select a stock ticker from the dropdown
2. Enter investment amount and time horizon
3. View interactive charts and calculations
4. Get AI-powered analysis and recommendations
5. Add new tickers with the automated pipeline

## 🛠️ Development
The app uses a modular architecture with separate components for:
- **Data ingestion and processing** (`Ingestion.py`)
- **Shared feature engineering** (`feature_engineering.py`) - eliminates code duplication
- **Machine learning model training** (`ModelBuilding.py`)
- **Prediction generation** (`predictions.py`)
- **DAG-based pipeline orchestration** (`ticker_dag.py`)
- **AI analysis and summarization** (`stock_summarizer.py`)

### 🎯 Recent Improvements
- **Eliminated code duplication**: Created shared `feature_engineering.py` module
- **Consistent calculations**: Same technical indicators across all components
- **Better maintainability**: Single source of truth for feature engineering
- **Enhanced UI**: Modern, professional design with color-coded metrics