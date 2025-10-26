"""
Stock Summarizer using OpenAI API to analyze stock behavior and provide structured insights.
"""

import os
import json
from datetime import datetime, date
from typing import Dict, Any
import pandas as pd
import duckdb
from openai import OpenAI
from dotenv import load_dotenv

from stock_analysis_models import StockAnalysisResponse

# Load environment variables
load_dotenv()

# Database path
DB_PATH = "data/duckdb/investo.duckdb"


class StockSummarizer:
    """Stock summarizer using OpenAI API for comprehensive analysis."""
    
    def __init__(self):
        """Initialize the summarizer with OpenAI client."""
        # Load environment variables from .env file
        load_dotenv()
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not self.client.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
    
    def get_stock_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive stock data for analysis."""
        con = duckdb.connect(DB_PATH, read_only=True)
        
        try:
            # Get current price and basic info
            current_price_query = """
            WITH last AS (
              SELECT ticker, MAX(date) AS d FROM prices GROUP BY ticker
            )
            SELECT p.adj_close AS current_price, p.date AS last_date
            FROM prices p
            JOIN last l ON p.ticker = l.ticker AND p.date = l.d
            WHERE p.ticker = ?
            """
            current_data = con.execute(current_price_query, [ticker]).df()
            
            # Get historical data (last 2 years)
            hist_query = """
            SELECT date, adj_close, high, low, volume
            FROM prices
            WHERE ticker = ? AND date >= CURRENT_DATE - INTERVAL '2 years'
            ORDER BY date
            """
            hist_data = con.execute(hist_query, [ticker]).df()
            
            # Get prediction data
            pred_query = """
            WITH latest AS (
              SELECT ticker, MAX(as_of_ts) AS asof
              FROM predictions
              WHERE ticker = ?
              GROUP BY ticker
            )
            SELECT p.pred_date, p.pred_price
            FROM predictions p
            JOIN latest l ON p.ticker = ? AND p.ticker = l.ticker AND p.as_of_ts = l.asof
            ORDER BY p.pred_date
            LIMIT 30
            """
            pred_data = con.execute(pred_query, [ticker, ticker]).df()
            
            # Get features for technical analysis
            features_query = """
            SELECT date, rsi_14, macd, macd_signal, bb_upper_20, bb_lower_20, 
                   sma_20, ema_12, ema_26, vol_21
            FROM features
            WHERE ticker = ? AND date >= CURRENT_DATE - INTERVAL '1 year'
            ORDER BY date DESC
            LIMIT 30
            """
            features_data = con.execute(features_query, [ticker]).df()
            
            return {
                "current_price": float(current_data["current_price"].iloc[0]) if not current_data.empty else None,
                "last_date": str(current_data["last_date"].iloc[0]) if not current_data.empty else None,
                "historical_data": hist_data.to_dict('records') if not hist_data.empty else [],
                "prediction_data": pred_data.to_dict('records') if not pred_data.empty else [],
                "features_data": features_data.to_dict('records') if not features_data.empty else []
            }
            
        finally:
            con.close()
    
    def calculate_technical_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical metrics from the data."""
        if not data["historical_data"]:
            return {}
        
        df = pd.DataFrame(data["historical_data"])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate basic metrics
        current_price = data["current_price"]
        if not current_price:
            return {}
        
        # Price change metrics
        price_52w_high = df['high'].max()
        price_52w_low = df['low'].min()
        price_change_1m = ((current_price - df['adj_close'].iloc[-30]) / df['adj_close'].iloc[-30] * 100) if len(df) >= 30 else 0
        price_change_3m = ((current_price - df['adj_close'].iloc[-90]) / df['adj_close'].iloc[-90] * 100) if len(df) >= 90 else 0
        price_change_1y = ((current_price - df['adj_close'].iloc[0]) / df['adj_close'].iloc[0] * 100) if len(df) > 0 else 0
        
        # Volatility
        returns = df['adj_close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) * 100  # Annualized volatility
        
        # Volume analysis
        avg_volume = df['volume'].mean()
        recent_volume = df['volume'].tail(5).mean()
        volume_trend = "increasing" if recent_volume > avg_volume else "decreasing"
        
        return {
            "current_price": current_price,
            "price_52w_high": price_52w_high,
            "price_52w_low": price_52w_low,
            "price_change_1m": price_change_1m,
            "price_change_3m": price_change_3m,
            "price_change_1y": price_change_1y,
            "volatility": volatility,
            "volume_trend": volume_trend,
            "avg_volume": avg_volume,
            "recent_volume": recent_volume
        }
    
    def create_analysis_prompt(self, ticker: str, data: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for OpenAI analysis."""
        
        # Prepare data summary
        current_price = data.get("current_price", "N/A")
        last_date = data.get("last_date", "N/A")
        
        # Historical data summary
        hist_data = data.get("historical_data", [])
        if hist_data:
            hist_df = pd.DataFrame(hist_data)
            price_range = f"${hist_df['adj_close'].min():.2f} - ${hist_df['adj_close'].max():.2f}"
            recent_trend = "upward" if len(hist_df) >= 2 and hist_df['adj_close'].iloc[-1] > hist_df['adj_close'].iloc[-2] else "downward"
        else:
            price_range = "N/A"
            recent_trend = "N/A"
        
        # Prediction data summary
        pred_data = data.get("prediction_data", [])
        if pred_data:
            pred_df = pd.DataFrame(pred_data)
            pred_trend = "upward" if len(pred_df) >= 2 and pred_df['pred_price'].iloc[-1] > pred_df['pred_price'].iloc[0] else "downward"
            pred_change = ((pred_df['pred_price'].iloc[-1] - current_price) / current_price * 100) if current_price else 0
        else:
            pred_trend = "N/A"
            pred_change = 0
        
        prompt = f"""
        You are a professional financial analyst with expertise in stock analysis, risk assessment, and investment strategy. 
        Analyze the following stock data for {ticker} and provide a comprehensive, structured analysis.

        STOCK DATA:
        - Ticker: {ticker}
        - Current Price: ${current_price}
        - Last Updated: {last_date}
        - 52-Week Price Range: {price_range}
        - Recent Trend: {recent_trend}
        - Predicted Trend: {pred_trend}
        - Predicted Price Change: {pred_change:.2f}%

        TECHNICAL METRICS:
        - 1-Month Change: {metrics.get('price_change_1m', 0):.2f}%
        - 3-Month Change: {metrics.get('price_change_3m', 0):.2f}%
        - 1-Year Change: {metrics.get('price_change_1y', 0):.2f}%
        - Volatility: {metrics.get('volatility', 0):.2f}%
        - Volume Trend: {metrics.get('volume_trend', 'N/A')}

        Please provide a comprehensive analysis covering:

        1. STOCK PERFORMANCE:
           - Current trend analysis (upward, downward, sideways)
           - Volatility assessment (low, moderate, high)
           - Price momentum (strong, weak, mixed)
           - Key support and resistance levels

        2. RISK ASSESSMENT:
           - Overall risk level (Low, Medium, High, Very High)
           - Key risk factors affecting this stock
           - Risk score from 1-10
           - Diversification advice

        3. INVESTMENT TIMING:
           - Optimal entry time
           - Optimal exit time
           - Recommended holding period
           - Current market conditions

        4. TECHNICAL ANALYSIS:
           - Key technical indicators and their signals
           - Identified chart patterns
           - Volume analysis
           - Trend strength

        5. FUNDAMENTAL ANALYSIS:
           - Company health assessment
           - Growth prospects
           - Competitive position
           - Valuation assessment

        6. INVESTMENT RECOMMENDATIONS:
           - Primary recommendation (Buy, Hold, Sell)
           - Alternative strategies
           - Portfolio allocation percentage
           - Stop-loss and target price suggestions

        7. MARKET OUTLOOK:
           - Sector outlook
           - Market sentiment (Bullish, Bearish, Neutral, Volatile)
           - External factors
           - Economic indicators

        Provide your analysis in a professional, data-driven manner. Be specific about numbers, percentages, and timeframes. 
        Include both opportunities and risks. Make sure your recommendations are actionable and well-reasoned.

        IMPORTANT: Return ONLY a valid JSON object with the exact nested structure shown below. Do not include any text before or after the JSON.
        The JSON must have all the nested objects (performance, risk_assessment, etc.) as shown.
        
        CRITICAL: Use these EXACT values for enums:
        - holding_period: "Short-term (1-6 months)", "Medium-term (6 months - 2 years)", or "Long-term (2+ years)"
        - risk_level: "Low", "Medium", "High", or "Very High"
        - market_sentiment: "Bullish", "Bearish", "Neutral", or "Volatile"

        Format your response as a JSON object with this EXACT structure:
        {{
            "summary": "Executive summary text",
            "performance": {{
                "current_trend": "upward/downward/sideways",
                "volatility_level": "low/moderate/high",
                "momentum": "strong/weak/mixed",
                "support_resistance": "description"
            }},
            "risk_assessment": {{
                "risk_level": "Low/Medium/High/Very High",
                "risk_factors": ["factor1", "factor2"],
                "risk_score": 5,
                "diversification_advice": "advice text"
            }},
            "investment_timing": {{
                "optimal_entry_time": "description",
                "optimal_exit_time": "description",
                "holding_period": "Short-term (1-6 months)",
                "market_conditions": "description"
            }},
            "technical_analysis": {{
                "key_indicators": ["indicator1", "indicator2"],
                "chart_patterns": ["pattern1", "pattern2"],
                "volume_analysis": "description",
                "trend_strength": "strong/moderate/weak"
            }},
            "fundamental_analysis": {{
                "company_health": "description",
                "growth_prospects": "description",
                "competitive_position": "description",
                "valuation_assessment": "description"
            }},
            "recommendations": {{
                "primary_recommendation": "Buy/Hold/Sell",
                "alternative_strategies": ["strategy1", "strategy2"],
                "portfolio_allocation": "5-10%",
                "stop_loss_suggestion": "suggestion",
                "target_price": "target"
            }},
            "market_outlook": {{
                "sector_outlook": "description",
                "market_sentiment": "Bullish/Bearish/Neutral/Volatile",
                "external_factors": ["factor1", "factor2"],
                "economic_indicators": "description"
            }},
            "key_insights": ["insight1", "insight2"],
            "warnings": ["warning1", "warning2"],
            "confidence_score": 7
        }}
        """
        
        return prompt
    
    def analyze_stock(self, ticker: str) -> StockAnalysisResponse:
        """Perform comprehensive stock analysis using OpenAI."""
        
        # Get stock data
        data = self.get_stock_data(ticker)
        if not data["current_price"]:
            raise ValueError(f"No data available for ticker {ticker}")
        
        # Calculate technical metrics
        metrics = self.calculate_technical_metrics(data)
        
        # Create analysis prompt
        prompt = self.create_analysis_prompt(ticker, data, metrics)
        
        # Get analysis from OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional financial analyst. Provide detailed, accurate stock analysis in JSON format. Return ONLY valid JSON without any additional text or formatting."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON from the response
            try:
                # Look for JSON in the response
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = analysis_text[start_idx:end_idx]
                    analysis_dict = json.loads(json_str)
                    
                    # Debug: Print the structure we received
                    print(f"DEBUG: Received keys: {list(analysis_dict.keys())}")
                    
                else:
                    raise ValueError("No JSON found in response")
                
                # Ensure the response has the correct nested structure
                analysis_dict = self._flatten_response(analysis_dict)
                
            except (json.JSONDecodeError, ValueError) as e:
                # If JSON parsing fails, create a structured response from the text
                print(f"DEBUG: JSON parsing failed: {e}")
                analysis_dict = self._parse_text_response(analysis_text, ticker)
            
            # Add current date
            analysis_dict["analysis_date"] = datetime.now().strftime("%Y-%m-%d")
            analysis_dict["ticker"] = ticker
            
            # Create Pydantic model
            return StockAnalysisResponse(**analysis_dict)
            
        except Exception as e:
            raise Exception(f"Error analyzing stock {ticker}: {str(e)}")
    
    def _flatten_response(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure response structure matches Pydantic model expectations."""
        # The response should already be in the correct nested structure
        # If it's flat, we need to restructure it
        if 'performance' not in response_dict and 'summary' in response_dict:
            # This means we got a flat response, need to use fallback
            return self._parse_text_response("", response_dict.get('ticker', 'UNKNOWN'))
        
        # If it's already nested, normalize enum values
        response_dict = self._normalize_enum_values(response_dict)
        
        return response_dict
    
    def _normalize_enum_values(self, response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize enum values to match Pydantic model expectations."""
        # Handle holding_period enum
        if 'investment_timing' in response_dict and isinstance(response_dict['investment_timing'], dict):
            holding_period = response_dict['investment_timing'].get('holding_period', '')
            if holding_period:
                # Map common variations to exact enum values
                if 'short' in holding_period.lower():
                    response_dict['investment_timing']['holding_period'] = 'Short-term (1-6 months)'
                elif 'medium' in holding_period.lower():
                    response_dict['investment_timing']['holding_period'] = 'Medium-term (6 months - 2 years)'
                elif 'long' in holding_period.lower():
                    response_dict['investment_timing']['holding_period'] = 'Long-term (2+ years)'
        
        # Handle risk_level enum
        if 'risk_assessment' in response_dict and isinstance(response_dict['risk_assessment'], dict):
            risk_level = response_dict['risk_assessment'].get('risk_level', '')
            if risk_level:
                # Map common variations to exact enum values
                if 'low' in risk_level.lower():
                    response_dict['risk_assessment']['risk_level'] = 'Low'
                elif 'medium' in risk_level.lower():
                    response_dict['risk_assessment']['risk_level'] = 'Medium'
                elif 'high' in risk_level.lower() and 'very' not in risk_level.lower():
                    response_dict['risk_assessment']['risk_level'] = 'High'
                elif 'very' in risk_level.lower() and 'high' in risk_level.lower():
                    response_dict['risk_assessment']['risk_level'] = 'Very High'
        
        # Handle market_sentiment enum
        if 'market_outlook' in response_dict and isinstance(response_dict['market_outlook'], dict):
            sentiment = response_dict['market_outlook'].get('market_sentiment', '')
            if sentiment:
                # Map common variations to exact enum values
                if 'bull' in sentiment.lower():
                    response_dict['market_outlook']['market_sentiment'] = 'Bullish'
                elif 'bear' in sentiment.lower():
                    response_dict['market_outlook']['market_sentiment'] = 'Bearish'
                elif 'neutral' in sentiment.lower():
                    response_dict['market_outlook']['market_sentiment'] = 'Neutral'
                elif 'volatil' in sentiment.lower():
                    response_dict['market_outlook']['market_sentiment'] = 'Volatile'
        
        return response_dict
    
    def _parse_text_response(self, text: str, ticker: str) -> Dict[str, Any]:
        """Parse text response into structured format if JSON parsing fails."""
        return {
            "summary": f"Analysis completed for {ticker} based on available data. AI analysis temporarily unavailable.",
            "performance": {
                "current_trend": "Mixed",
                "volatility_level": "Moderate",
                "momentum": "Neutral",
                "support_resistance": "Standard levels"
            },
            "risk_assessment": {
                "risk_level": "Medium",
                "risk_factors": ["Market volatility", "Economic uncertainty"],
                "risk_score": 5,
                "diversification_advice": "Consider diversifying across sectors"
            },
            "investment_timing": {
                "optimal_entry_time": "Gradual entry recommended",
                "optimal_exit_time": "Monitor for exit signals",
                "holding_period": "Medium-term (6 months - 2 years)",
                "market_conditions": "Mixed market conditions"
            },
            "technical_analysis": {
                "key_indicators": ["RSI", "MACD", "Moving averages"],
                "chart_patterns": ["Standard patterns observed"],
                "volume_analysis": "Average volume levels",
                "trend_strength": "Moderate"
            },
            "fundamental_analysis": {
                "company_health": "Stable",
                "growth_prospects": "Moderate growth expected",
                "competitive_position": "Competitive in sector",
                "valuation_assessment": "Fairly valued"
            },
            "recommendations": {
                "primary_recommendation": "Hold with monitoring",
                "alternative_strategies": ["Dollar-cost averaging", "Sector rotation"],
                "portfolio_allocation": "5-10%",
                "stop_loss_suggestion": "Set at 10% below entry",
                "target_price": "Monitor for 15-20% gains"
            },
            "market_outlook": {
                "sector_outlook": "Neutral to positive",
                "market_sentiment": "Neutral",
                "external_factors": ["Economic policy", "Market sentiment"],
                "economic_indicators": "Mixed signals"
            },
            "key_insights": ["Regular monitoring recommended", "Consider risk management"],
            "warnings": ["Market volatility", "Economic uncertainty"],
            "confidence_score": 6
        }
    
    def _create_fallback_response(self, ticker: str) -> Dict[str, Any]:
        """Create a proper fallback response with all required fields."""
        return self._parse_text_response("", ticker)


def get_stock_analysis(ticker: str) -> StockAnalysisResponse:
    """Convenience function to get stock analysis."""
    summarizer = StockSummarizer()
    return summarizer.analyze_stock(ticker)
