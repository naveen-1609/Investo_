#!/usr/bin/env python3
"""
Ticker Addition DAG - Complete Pipeline Orchestrator

This module orchestrates the complete pipeline for adding new tickers:
1. Data Ingestion (download historical data)
2. Feature Engineering (compute technical indicators)
3. Model Retraining (retrain global LSTM with new ticker)
4. Prediction Generation (generate future predictions)
5. Database Updates (update ticker lists and metadata)

The DAG ensures proper dependency management and error handling.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import duckdb
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Import existing modules
from Ingestion import main as ingestion_main, ensure_duckdb_initialized
from ModelBuilding import train_global
from predictions import main as prediction_main

# ------------------------------
# Configuration
# ------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DUCKDB_PATH = DATA_DIR / "duckdb" / "investo.duckdb"
TICKERS_FILE = ROOT / "tickers.json"

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ticker_dag")

# ------------------------------
# DAG State Management
# ------------------------------
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskResult:
    task_name: str
    status: TaskStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None

class TickerAdditionDAG:
    """DAG orchestrator for adding new tickers to the investment system."""
    
    def __init__(self):
        self.tasks = []
        self.results = {}
        self.current_ticker = None
        
    def add_task(self, task_name: str, task_func, dependencies: List[str] = None):
        """Add a task to the DAG."""
        self.tasks.append({
            'name': task_name,
            'function': task_func,
            'dependencies': dependencies or [],
            'status': TaskStatus.PENDING
        })
    
    def can_run_task(self, task_name: str) -> bool:
        """Check if a task can be run based on its dependencies."""
        task = next((t for t in self.tasks if t['name'] == task_name), None)
        if not task:
            return False
            
        for dep in task['dependencies']:
            if dep not in self.results or self.results[dep].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def run_task(self, task_name: str) -> TaskResult:
        """Run a single task."""
        task = next((t for t in self.tasks if t['name'] == task_name), None)
        if not task:
            return TaskResult(task_name, TaskStatus.FAILED, error_message="Task not found")
        
        if not self.can_run_task(task_name):
            return TaskResult(task_name, TaskStatus.SKIPPED, error_message="Dependencies not met")
        
        log.info(f"Starting task: {task_name}")
        result = TaskResult(task_name, TaskStatus.RUNNING, start_time=datetime.now())
        
        try:
            # Execute the task function
            output = task['function'](self.current_ticker, self.results)
            result.status = TaskStatus.COMPLETED
            result.end_time = datetime.now()
            result.output_data = output
            log.info(f"Completed task: {task_name}")
            
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.end_time = datetime.now()
            result.error_message = str(e)
            log.error(f"Failed task: {task_name} - {str(e)}")
        
        return result
    
    def run_pipeline(self, ticker: str) -> Dict[str, TaskResult]:
        """Run the complete pipeline for adding a new ticker."""
        self.current_ticker = ticker
        self.results = {}
        
        log.info(f"Starting pipeline for ticker: {ticker}")
        
        # Define the pipeline tasks
        self._setup_pipeline_tasks()
        
        # Run tasks in dependency order
        completed_tasks = set()
        max_iterations = len(self.tasks) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(completed_tasks) < len(self.tasks) and iteration < max_iterations:
            iteration += 1
            progress_made = False
            
            for task in self.tasks:
                task_name = task['name']
                if task_name in completed_tasks:
                    continue
                
                if self.can_run_task(task_name):
                    result = self.run_task(task_name)
                    self.results[task_name] = result
                    completed_tasks.add(task_name)
                    progress_made = True
                    
                    # If task failed, stop the pipeline
                    if result.status == TaskStatus.FAILED:
                        log.error(f"Pipeline failed at task: {task_name}")
                        break
            
            if not progress_made:
                log.error("Pipeline stuck - no progress made")
                break
        
        # Check if all tasks completed successfully
        failed_tasks = [name for name, result in self.results.items() 
                       if result.status == TaskStatus.FAILED]
        
        if failed_tasks:
            log.error(f"Pipeline failed. Failed tasks: {failed_tasks}")
        else:
            log.info("Pipeline completed successfully!")
        
        return self.results

    def _setup_pipeline_tasks(self):
        """Setup the pipeline tasks with their dependencies."""
        self.tasks = []  # Reset tasks
        
        # Task 1: Validate ticker
        self.add_task("validate_ticker", self._validate_ticker)
        
        # Task 2: Update ticker list
        self.add_task("update_ticker_list", self._update_ticker_list, ["validate_ticker"])
        
        # Task 3: Data ingestion
        self.add_task("data_ingestion", self._run_ingestion, ["update_ticker_list"])
        
        # Task 4: Feature engineering (handled by ingestion)
        self.add_task("feature_engineering", self._verify_features, ["data_ingestion"])
        
        # Task 5: Model retraining
        self.add_task("model_retraining", self._run_training, ["feature_engineering"])
        
        # Task 6: Generate predictions
        self.add_task("generate_predictions", self._run_predictions, ["model_retraining"])
        
        # Task 7: Verify integration
        self.add_task("verify_integration", self._verify_integration, ["generate_predictions"])

    # Task implementations
    def _validate_ticker(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Validate that the ticker is valid and not already in the system."""
        log.info(f"Validating ticker: {ticker}")
        
        # Check if ticker already exists
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            existing_tickers = con.execute("SELECT DISTINCT ticker FROM prices").df()
            if ticker in existing_tickers['ticker'].values:
                raise ValueError(f"Ticker {ticker} already exists in the system")
        finally:
            con.close()
        
        # Basic validation (you could add more sophisticated validation here)
        if not ticker or len(ticker) < 1 or len(ticker) > 10:
            raise ValueError(f"Invalid ticker format: {ticker}")
        
        return {"ticker": ticker, "validated": True}
    
    def _update_ticker_list(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Update the tickers.json file with the new ticker."""
        log.info(f"Updating ticker list with: {ticker}")
        
        # Load current tickers
        if TICKERS_FILE.exists():
            with open(TICKERS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {"tickers": []}
        
        # Add new ticker if not already present
        if ticker not in data["tickers"]:
            data["tickers"].append(ticker)
            data["tickers"].sort()  # Keep sorted
            
            # Save updated tickers
            with open(TICKERS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            
            log.info(f"Added {ticker} to ticker list")
        else:
            log.info(f"Ticker {ticker} already in list")
        
        return {"tickers": data["tickers"], "added": ticker}
    
    def _run_ingestion(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Run data ingestion for the new ticker."""
        log.info(f"Running data ingestion for: {ticker}")
        
        # Ensure database is initialized
        ensure_duckdb_initialized()
        
        # Run ingestion (this will process all tickers in tickers.json)
        try:
            ingestion_main(str(TICKERS_FILE), replace_existing=False)
            log.info("Data ingestion completed successfully")
        except Exception as e:
            log.error(f"Data ingestion failed: {str(e)}")
            raise
        
        # Verify the ticker was added
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            ticker_data = con.execute(
                "SELECT COUNT(*) as count FROM prices WHERE ticker = ?", 
                [ticker]
            ).df()
            
            if ticker_data['count'].iloc[0] == 0:
                raise ValueError(f"No data found for ticker {ticker} after ingestion")
            
            log.info(f"Verified {ticker} data in database")
        finally:
            con.close()
        
        return {"ticker": ticker, "ingestion_completed": True}
    
    def _verify_features(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Verify that features were computed for the new ticker."""
        log.info(f"Verifying features for: {ticker}")
        
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            feature_count = con.execute(
                "SELECT COUNT(*) as count FROM features WHERE ticker = ?", 
                [ticker]
            ).df()
            
            if feature_count['count'].iloc[0] == 0:
                raise ValueError(f"No features found for ticker {ticker}")
            
            log.info(f"Verified features for {ticker}")
        finally:
            con.close()
        
        return {"ticker": ticker, "features_verified": True}
    
    def _run_training(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Run model retraining with the new ticker included."""
        log.info(f"Running model retraining with: {ticker}")
        
        try:
            train_global(
                lookback=60, horizon=30, hidden=64, layers=2,
                embed_dim=8, dropout=0.2, epochs=20, batch=64,
                lr=1e-3, val_ratio=0.2
            )
            log.info("Model retraining completed successfully")
        except Exception as e:
            log.error(f"Model retraining failed: {str(e)}")
            raise
        
        return {"ticker": ticker, "training_completed": True}
    
    def _run_predictions(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Generate predictions for all tickers including the new one."""
        log.info(f"Generating predictions including: {ticker}")
        
        try:
            # Generate predictions for 2 years ahead
            prediction_main(years=2)
            log.info("Prediction generation completed successfully")
        except Exception as e:
            log.error(f"Prediction generation failed: {str(e)}")
            raise
        
        return {"ticker": ticker, "predictions_completed": True}
    
    def _verify_integration(self, ticker: str, results: Dict[str, TaskResult]) -> Dict[str, Any]:
        """Verify that the new ticker is fully integrated into the system."""
        log.info(f"Verifying integration for: {ticker}")
        
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            # Check predictions exist
            pred_count = con.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE ticker = ?", 
                [ticker]
            ).df()
            
            if pred_count['count'].iloc[0] == 0:
                raise ValueError(f"No predictions found for ticker {ticker}")
            
            log.info(f"Verified integration for {ticker}")
        finally:
            con.close()
        
        return {"ticker": ticker, "integration_verified": True}

# ------------------------------
# Convenience Functions
# ------------------------------
def add_new_ticker(ticker: str) -> Dict[str, TaskResult]:
    """Add a new ticker to the system through the complete pipeline."""
    dag = TickerAdditionDAG()
    return dag.run_pipeline(ticker)

def get_available_tickers() -> List[str]:
    """Get list of available tickers from the database."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        tickers_df = con.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").df()
        return tickers_df['ticker'].tolist()
    finally:
        con.close()

def get_pipeline_status(ticker: str) -> Dict[str, Any]:
    """Get the current status of a ticker in the pipeline."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        # Check if ticker exists in prices
        price_count = con.execute(
            "SELECT COUNT(*) as count FROM prices WHERE ticker = ?", 
            [ticker]
        ).df()['count'].iloc[0]
        
        # Check if ticker exists in features
        feature_count = con.execute(
            "SELECT COUNT(*) as count FROM features WHERE ticker = ?", 
            [ticker]
        ).df()['count'].iloc[0]
        
        # Check if ticker exists in predictions
        pred_count = con.execute(
            "SELECT COUNT(*) as count FROM predictions WHERE ticker = ?", 
            [ticker]
        ).df()['count'].iloc[0]
        
        return {
            "ticker": ticker,
            "has_price_data": price_count > 0,
            "has_features": feature_count > 0,
            "has_predictions": pred_count > 0,
            "is_fully_integrated": all([price_count > 0, feature_count > 0, pred_count > 0])
        }
    finally:
        con.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python ticker_dag.py <TICKER>")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    print(f"Adding ticker: {ticker}")
    
    results = add_new_ticker(ticker)
    
    # Print results
    for task_name, result in results.items():
        print(f"{task_name}: {result.status.value}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
