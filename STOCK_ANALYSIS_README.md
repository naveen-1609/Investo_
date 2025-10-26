# Stock Analysis with AI

This module provides comprehensive AI-powered stock analysis using OpenAI's GPT-4 model. It analyzes stock behavior, risk assessment, investment timing, and provides structured recommendations.

## Features

### 🤖 AI-Powered Analysis
- **Comprehensive Stock Analysis**: Uses OpenAI GPT-4 to analyze stock behavior
- **Structured Output**: Pydantic models ensure consistent, structured responses
- **Real-time Data Integration**: Combines historical data, predictions, and technical indicators

### 📊 Analysis Components

1. **Stock Performance**
   - Current trend analysis (upward, downward, sideways)
   - Volatility assessment (low, moderate, high)
   - Price momentum (strong, weak, mixed)
   - Support and resistance levels

2. **Risk Assessment**
   - Overall risk level (Low, Medium, High, Very High)
   - Risk score (1-10 scale)
   - Key risk factors identification
   - Diversification advice

3. **Investment Timing**
   - Optimal entry and exit times
   - Recommended holding period
   - Market conditions analysis

4. **Technical Analysis**
   - Key technical indicators (RSI, MACD, Moving averages)
   - Chart pattern identification
   - Volume analysis
   - Trend strength assessment

5. **Fundamental Analysis**
   - Company health assessment
   - Growth prospects
   - Competitive position
   - Valuation assessment

6. **Investment Recommendations**
   - Primary recommendation (Buy, Hold, Sell)
   - Alternative strategies
   - Portfolio allocation percentage
   - Stop-loss and target price suggestions

7. **Market Outlook**
   - Sector outlook
   - Market sentiment (Bullish, Bearish, Neutral, Volatile)
   - External factors affecting the stock
   - Economic indicators

## Setup

### 1. Environment Variables
Create a `.env` file in the project root with your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 2. Dependencies
The following packages are required and already included in `pyproject.toml`:
- `openai>=2.6.0`
- `pydantic>=2.0.0`
- `python-dotenv>=1.0.0`

### 3. Data Requirements
Ensure you have:
- Historical price data in the `prices` table
- Prediction data in the `predictions` table
- Technical features in the `features` table

## Usage

### In the Investment Calculator
1. Select a stock from the dropdown
2. Click the "Analyze Stock" button
3. Wait for the AI analysis to complete
4. Review the comprehensive analysis in organized tabs

### Programmatic Usage
```python
from stock_summarizer import get_stock_analysis

# Get comprehensive analysis for a stock
analysis = get_stock_analysis("AAPL")

# Access specific components
print(f"Risk Level: {analysis.risk_assessment.risk_level}")
print(f"Recommendation: {analysis.recommendations.primary_recommendation}")
print(f"Confidence Score: {analysis.confidence_score}/10")
```

## Data Sources

The analysis combines multiple data sources:

1. **Historical Data**: Last 2 years of price and volume data
2. **Prediction Data**: ML model predictions for future prices
3. **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
4. **Market Data**: Current prices, volatility, volume trends

## Output Structure

The analysis returns a structured `StockAnalysisResponse` object with:

- **Executive Summary**: High-level overview
- **Performance Metrics**: Current trend, volatility, momentum
- **Risk Assessment**: Risk level, factors, and score
- **Investment Timing**: Entry/exit recommendations
- **Technical Analysis**: Indicators and patterns
- **Fundamental Analysis**: Company health and prospects
- **Recommendations**: Actionable investment advice
- **Market Outlook**: Sector and economic factors
- **Key Insights**: Important takeaways
- **Warnings**: Risk factors and cautions
- **Confidence Score**: Analysis reliability (1-10)

## Error Handling

The system includes robust error handling:
- Graceful fallback if OpenAI API is unavailable
- Data validation using Pydantic models
- Clear error messages for missing data
- Fallback analysis if JSON parsing fails

## Performance

- **Caching**: Results are cached to avoid redundant API calls
- **Efficient Queries**: Optimized database queries for data retrieval
- **Async Support**: Ready for future async implementation
- **Rate Limiting**: Respects OpenAI API rate limits

## Security

- **API Key Protection**: Environment variable storage
- **Data Privacy**: No sensitive data sent to external services
- **Input Validation**: Pydantic models ensure data integrity
- **Error Sanitization**: Prevents information leakage in error messages

## Future Enhancements

- **Multi-model Support**: Integration with other AI models
- **Real-time Updates**: Live analysis updates
- **Portfolio Analysis**: Multi-stock portfolio analysis
- **Custom Prompts**: User-defined analysis criteria
- **Export Options**: PDF/Excel report generation
