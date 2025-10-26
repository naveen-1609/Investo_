"""
Pydantic models for structured stock analysis responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level enumeration."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"


class MarketSentiment(str, Enum):
    """Market sentiment enumeration."""
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"
    VOLATILE = "Volatile"


class TimeHorizon(str, Enum):
    """Investment time horizon enumeration."""
    SHORT_TERM = "Short-term (1-6 months)"
    MEDIUM_TERM = "Medium-term (6 months - 2 years)"
    LONG_TERM = "Long-term (2+ years)"


class StockPerformance(BaseModel):
    """Stock performance metrics."""
    current_trend: str = Field(description="Current price trend (upward, downward, sideways)")
    volatility_level: str = Field(description="Volatility assessment (low, moderate, high)")
    momentum: str = Field(description="Price momentum (strong, weak, mixed)")
    support_resistance: str = Field(description="Key support and resistance levels")


class RiskAssessment(BaseModel):
    """Risk assessment for the stock."""
    risk_level: RiskLevel = Field(description="Overall risk level")
    risk_factors: List[str] = Field(description="List of key risk factors")
    risk_score: int = Field(ge=1, le=10, description="Risk score from 1 (lowest) to 10 (highest)")
    diversification_advice: str = Field(description="Diversification recommendations")


class InvestmentTiming(BaseModel):
    """Investment timing recommendations."""
    optimal_entry_time: str = Field(description="Best time to enter the position")
    optimal_exit_time: str = Field(description="Best time to exit the position")
    holding_period: TimeHorizon = Field(description="Recommended holding period")
    market_conditions: str = Field(description="Current market conditions affecting timing")


class TechnicalAnalysis(BaseModel):
    """Technical analysis insights."""
    key_indicators: List[str] = Field(description="Key technical indicators and their signals")
    chart_patterns: List[str] = Field(description="Identified chart patterns")
    volume_analysis: str = Field(description="Volume analysis and implications")
    trend_strength: str = Field(description="Strength of current trend")


class FundamentalAnalysis(BaseModel):
    """Fundamental analysis insights."""
    company_health: str = Field(description="Overall company financial health")
    growth_prospects: str = Field(description="Growth prospects and outlook")
    competitive_position: str = Field(description="Competitive position in the market")
    valuation_assessment: str = Field(description="Current valuation assessment")


class InvestmentRecommendations(BaseModel):
    """Investment recommendations and suggestions."""
    primary_recommendation: str = Field(description="Primary investment recommendation")
    alternative_strategies: List[str] = Field(description="Alternative investment strategies")
    portfolio_allocation: str = Field(description="Recommended portfolio allocation percentage")
    stop_loss_suggestion: Optional[str] = Field(description="Stop-loss level suggestion")
    target_price: Optional[str] = Field(description="Target price for the investment")


class MarketOutlook(BaseModel):
    """Market outlook and external factors."""
    sector_outlook: str = Field(description="Outlook for the stock's sector")
    market_sentiment: MarketSentiment = Field(description="Overall market sentiment")
    external_factors: List[str] = Field(description="External factors affecting the stock")
    economic_indicators: str = Field(description="Relevant economic indicators")


class StockAnalysisResponse(BaseModel):
    """Complete stock analysis response."""
    ticker: str = Field(description="Stock ticker symbol")
    analysis_date: str = Field(description="Date of analysis")
    summary: str = Field(description="Executive summary of the stock analysis")
    
    # Core analysis components
    performance: StockPerformance = Field(description="Stock performance analysis")
    risk_assessment: RiskAssessment = Field(description="Risk assessment")
    investment_timing: InvestmentTiming = Field(description="Investment timing recommendations")
    technical_analysis: TechnicalAnalysis = Field(description="Technical analysis insights")
    fundamental_analysis: FundamentalAnalysis = Field(description="Fundamental analysis insights")
    recommendations: InvestmentRecommendations = Field(description="Investment recommendations")
    market_outlook: MarketOutlook = Field(description="Market outlook and external factors")
    
    # Additional insights
    key_insights: List[str] = Field(description="Key insights and takeaways")
    warnings: List[str] = Field(description="Important warnings and cautions")
    confidence_score: int = Field(ge=1, le=10, description="Confidence in analysis from 1-10")
