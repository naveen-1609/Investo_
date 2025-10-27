# 💰 Investo - Complete Documentation

## Overview

Investo is a comprehensive Streamlit-based web application that provides stock analysis, investment calculations, AI-powered insights, and dynamic ticker management. The app combines machine learning predictions, technical analysis, and OpenAI integration to deliver a complete investment analysis platform.

## Architecture

The app follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web interface (`investment_calculator.py`)
- **Data Pipeline**: Ingestion, model training, and prediction modules
- **AI Integration**: OpenAI-powered stock analysis
- **Database**: DuckDB for efficient data storage
- **Orchestration**: DAG-based ticker management system

## File Structure & Functionality

### Core Application Files

#### 1. `investment_calculator.py` - Main Application
**Purpose**: The primary Streamlit web application that serves as the user interface.

**Key Features**:
- **Multi-tab Interface**: 
  - 📊 Investment Calculator: Main investment analysis and visualization
  - 🤖 AI Stock Analysis: OpenAI-powered stock insights
  - ➕ Add New Ticker: Dynamic ticker management system

- **Investment Calculator Tab**:
  - Interactive stock selection dropdown
  - Investment amount and timeframe inputs
  - Real-time price charts (70% of screen)
  - Investment calculation results (30% of screen)
  - Expected returns comparison table for other stocks

- **AI Stock Analysis Tab**:
  - OpenAI integration for comprehensive stock analysis
  - Technical indicators and market sentiment analysis
  - Investment recommendations and risk assessment

- **Add New Ticker Tab**:
  - Dropdown selection of available tickers
  - Complete pipeline execution (ingestion → training → prediction)
  - Real-time status monitoring
  - Automatic model retraining on global dataset

**Technical Implementation**:
```python
# Dynamic database path resolution
if os.path.exists("data/duckdb/investo.duckdb"):
    DB_PATH = "data/duckdb/investo.duckdb"
elif os.path.exists(".venv/data/duckdb/investo.duckdb"):
    DB_PATH = ".venv/data/duckdb/investo.duckdb"
else:
    script_dir = Path(__file__).parent
    DB_PATH = str(script_dir / "data" / "duckdb" / "investo.duckdb")
```

**Styling Features**:
- Color-coded return percentages (green for positive, red for negative)
- Professional table styling with alternating row colors
- Responsive layout with proper spacing and typography

#### 2. `ticker_dag.py` - Pipeline Orchestrator
**Purpose**: Manages the complete lifecycle of adding new tickers to the system.

**Key Functions**:
- `add_new_ticker(ticker)`: Orchestrates the full pipeline
- `get_pipeline_status()`: Returns current pipeline execution status
- `get_available_tickers()`: Retrieves list of available tickers

**Pipeline Flow**:
1. **Data Ingestion**: Downloads historical data for the new ticker
2. **Model Training**: Retrains the global LSTM model on all data
3. **Prediction Generation**: Creates predictions for the new ticker
4. **Database Update**: Stores predictions and updates ticker list

**Technical Implementation**:
```python
def add_new_ticker(ticker):
    """Add a new ticker through the complete pipeline"""
    try:
        # Step 1: Data ingestion
        ingestion_main(ticker)
        
        # Step 2: Model training
        train_global()
        
        # Step 3: Generate predictions
        prediction_main()
        
        return {"status": "success", "message": f"Ticker {ticker} added successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### Data Processing Pipeline

#### 3. `Ingestion.py` - Data Collection Module
**Purpose**: Handles data collection and database initialization.

**Key Functions**:
- `main(ticker)`: Downloads and processes stock data for a specific ticker
- `ensure_duckdb_initialized()`: Sets up database schema and tables
- Data validation and cleaning
- Historical price data collection

**Data Sources**:
- Yahoo Finance API for historical stock data
- Technical indicators calculation
- Data quality validation and error handling

#### 4. `ModelBuilding.py` - Machine Learning Module
**Purpose**: Handles LSTM model training and management.

**Key Functions**:
- `train_global()`: Trains the global LSTM model on all available data
- Feature engineering and data preprocessing
- Model persistence and versioning
- Cross-validation and performance metrics

**Model Architecture**:
- LSTM neural network for time series prediction
- Global training on entire dataset
- Scalable architecture for multiple tickers

#### 5. `predictions.py` - Prediction Generation
**Purpose**: Generates future price predictions using trained models.

**Key Functions**:
- `main()`: Generates predictions for all tickers
- Model loading and inference
- Prediction confidence intervals
- Results storage and formatting

**Prediction Features**:
- Multi-timeframe predictions
- Confidence scoring
- Risk assessment metrics

### AI Integration

#### 6. `stock_summarizer.py` - AI Analysis Module
**Purpose**: Integrates OpenAI API for comprehensive stock analysis.

**Key Classes**:
- `StockSummarizer`: Main class for AI-powered analysis

**Key Methods**:
- `fetch_stock_data()`: Retrieves current stock information
- `calculate_technical_indicators()`: Computes technical analysis metrics
- `create_analysis_prompt()`: Generates structured prompts for OpenAI
- `analyze_stock()`: Executes AI analysis and returns structured results

**AI Analysis Features**:
- Technical indicator interpretation
- Market sentiment analysis
- Investment recommendations
- Risk assessment
- Price target predictions

**Technical Implementation**:
```python
class StockSummarizer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def analyze_stock(self, ticker: str) -> StockAnalysis:
        # Fetch data, calculate indicators, create prompt, analyze
        pass
```

#### 7. `stock_analysis_models.py` - Data Models
**Purpose**: Defines Pydantic models for structured data handling.

**Key Models**:
- `StockAnalysis`: Complete analysis result structure
- `TechnicalIndicators`: Technical analysis metrics
- `InvestmentRecommendation`: AI-generated recommendations
- `RiskAssessment`: Risk analysis results

**Benefits**:
- Type safety and validation
- Structured API responses
- Easy serialization/deserialization
- Clear data contracts

### Configuration & Data

#### 8. `tickers.json` - Ticker Configuration
**Purpose**: Maintains the list of available stock tickers.

**Structure**:
```json
{
  "tickers": ["AAPL", "AMZN", "GOOGL", "JPM", "MSFT", "NVDA"]
}
```

**Usage**:
- Dropdown population in the UI
- Pipeline execution triggers
- Dynamic ticker management

#### 9. Database Schema (`data/duckdb/investo.duckdb`)
**Purpose**: Centralized data storage using DuckDB.

**Tables**:
- `prices`: Historical stock price data
- `features`: Engineered features for ML models
- `predictions`: Generated price predictions
- `predictions_hive`: Partitioned predictions for performance

**Benefits**:
- High-performance analytical queries
- Columnar storage for efficient data access
- SQL interface for complex operations
- Local file-based storage

## Application Workflow

### 1. Investment Analysis Workflow
```
User Input → Stock Selection → Data Retrieval → Chart Generation → 
Investment Calculation → Results Display → Comparison Table
```

### 2. AI Analysis Workflow
```
Ticker Selection → Data Fetching → Technical Indicators → 
Prompt Generation → OpenAI Analysis → Structured Response → Display
```

### 3. New Ticker Addition Workflow
```
Ticker Selection → Data Ingestion → Model Retraining → 
Prediction Generation → Database Update → UI Refresh
```

## Key Features

### 1. Real-time Data Visualization
- Interactive Plotly charts
- 70/30 layout split for optimal viewing
- Responsive design for different screen sizes

### 2. Investment Calculations
- Future value projections
- Return percentage calculations
- Risk-adjusted metrics
- Comparative analysis across stocks

### 3. AI-Powered Insights
- OpenAI GPT integration
- Technical analysis interpretation
- Market sentiment analysis
- Investment recommendations

### 4. Dynamic Ticker Management
- Add new stocks to the system
- Automatic model retraining
- Pipeline status monitoring
- Real-time updates

### 5. Professional UI/UX
- Color-coded return indicators
- Professional table styling
- Intuitive navigation
- Responsive design

## Technical Stack

### Frontend
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualization
- **Pandas**: Data manipulation and styling

### Backend
- **Python**: Core programming language
- **DuckDB**: Analytical database
- **OpenAI API**: AI analysis integration

### Machine Learning
- **PyTorch**: LSTM model implementation
- **Scikit-learn**: Data preprocessing and scaling
- **Pandas**: Feature engineering

### Data Sources
- **Yahoo Finance**: Historical stock data
- **OpenAI GPT**: AI analysis and insights

## Deployment Configuration

### Required Files
- `requirements.txt`: Python dependencies
- `Procfile`: Heroku deployment configuration
- `runtime.txt`: Python version specification
- `.streamlit/config.toml`: Streamlit configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API access key
- Database connection settings
- Model file paths

## Performance Considerations

### 1. Database Optimization
- Columnar storage for analytical queries
- Partitioned tables for large datasets
- Efficient indexing strategies

### 2. Model Management
- Global model training for consistency
- Model versioning and persistence
- Efficient inference pipelines

### 3. Caching Strategy
- Streamlit caching for expensive operations
- Database query optimization
- API response caching

## Conclusion

The Investment Calculator App represents a comprehensive solution for stock analysis and investment decision-making. With its modular architecture, AI integration, and user-friendly interface, it provides both novice and experienced investors with powerful tools for market analysis and investment planning.

The application successfully combines traditional financial analysis with modern AI capabilities, creating a robust platform for investment research and decision-making.
