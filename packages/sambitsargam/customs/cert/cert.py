# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Sambit Sargam Ekalabya
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""
This module implements the Crypto-Enhanced AI Risk Assessment Tool (CERT).
It integrates AI-powered sentiment analysis, technical indicator checks, on-chain data integration,
and predictive forecasting to produce a comprehensive risk assessment report.
"""

import functools
import json
import requests
from typing import Any, Dict, Optional, Tuple, Callable

import openai
from pydantic import BaseModel

# Define the MechResponse type as a tuple (result, message, metadata, extra, api_keys)
MechResponse = Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]


def with_key_rotation(func: Callable) -> Callable:
    """
    Decorator to manage API key rotation on rate limit errors.
    The decorated function should return a 4-tuple; this decorator adds the api_keys as the fifth element.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> MechResponse:
        # Expecting api_keys to be provided in kwargs.
        api_keys = kwargs["api_keys"]
        retries_left: Dict[str, int] = api_keys.max_retries()  # assuming api_keys has a max_retries() method

        def execute() -> MechResponse:
            try:
                result = func(*args, **kwargs)
                return result + (api_keys,)
            except openai.RateLimitError as e:
                if retries_left.get("openai", 0) <= 0 and retries_left.get("openrouter", 0) <= 0:
                    raise e
                retries_left["openai"] -= 1
                retries_left["openrouter"] -= 1
                api_keys.rotate("openai")
                api_keys.rotate("openrouter")
                return execute()
            except Exception as e:
                return str(e), None, None, None, api_keys

        return execute()
    return wrapper


def error_response(msg: str) -> Tuple[str, None, None, None]:
    """Return an error mech response."""
    return msg, None, None, None


# --- CORE FUNCTIONS FOR CERT RISK ASSESSMENT --- #

def analyze_sentiment(project: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs sentiment analysis on the project's community data.
    In production, integrate social media sources and ML sentiment models.
    """
    # For demonstration, we assume a positive sentiment.
    return {
        "score": 0.8,
        "summary": f"Strong positive sentiment detected for project '{project}'."
    }


def fetch_technical_indicators(project: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves technical market indicators.
    In production, integrate with TAapi or another data provider.
    """
    return {
        "RSI": 40,
        "MACD": 1.2,
        "MovingAverages": {"50_day": 102.5, "200_day": 98.3}
    }


def fetch_onchain_data(contract_address: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves live on-chain metrics using the Etherscan API.
    
    Args:
        contract_address (str): Ethereum contract address.
        api_key (Optional[str]): Valid Etherscan API key.
    
    Returns:
        Dict[str, Any]: Contains transaction count and unique address count.
        
    Raises:
        ValueError: If an API key is not provided or if the API returns an error.
    """
    if api_key is None:
        raise ValueError("A valid Etherscan API key must be provided for on-chain data retrieval.")
    url = "https://api.etherscan.io/api"
    params = {
        "module": "account",
        "action": "txlist",
        "address": contract_address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "1":
        raise ValueError(f"Error fetching on-chain data: {data.get('message')}")
    txs = data.get("result", [])
    transaction_count = len(txs)
    unique_addresses = set()
    for tx in txs:
        if "from" in tx:
            unique_addresses.add(tx["from"])
        if "to" in tx:
            unique_addresses.add(tx["to"])
    return {
        "transaction_count": transaction_count,
        "unique_addresses_count": len(unique_addresses)
    }


def predictive_forecasting(project: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Provides a predictive forecast based on market data.
    In production, integrate advanced ML models and historical data analysis.
    """
    return {
        "predicted_growth": 0.15,
        "forecast_summary": f"Projected moderate growth for project '{project}' over the next 6 months."
    }


def calculate_risk_rating(
    sentiment_score: float,
    technical: Dict[str, Any],
    onchain: Dict[str, Any],
    forecast: Dict[str, Any]
) -> str:
    """
    Calculates an overall risk rating using simple heuristics.
    """
    if sentiment_score >= 0.75 and technical.get("RSI", 0) < 50:
        return "Low Risk"
    elif sentiment_score >= 0.5:
        return "Moderate Risk"
    else:
        return "High Risk"


def generate_recommendations(risk_rating: str) -> str:
    """
    Provides recommendations based on the risk rating.
    """
    if risk_rating == "Low Risk":
        return "Consider a larger allocation to this project."
    elif risk_rating == "Moderate Risk":
        return "Maintain a balanced portfolio with caution."
    else:
        return "Proceed with caution or reconsider investment."


class RiskAssessmentReport(BaseModel):
    project: str
    sentiment_score: float
    sentiment_summary: str
    technical_indicators: Dict[str, Any]
    onchain_metrics: Dict[str, Any]
    forecast: Dict[str, Any]
    risk_rating: str
    recommendations: str


def run_cert_assessment(
    project: str,
    contract_address: str,
    api_keys: Dict[str, Any]
) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any]:
    """
    Executes the CERT risk assessment and returns a JSON report along with a status message and metadata.
    """
    sentiment_result = analyze_sentiment(project, api_key=api_keys.get("openai"))
    technical_result = fetch_technical_indicators(project, api_key=api_keys.get("taapi"))
    onchain_result = fetch_onchain_data(contract_address, api_key=api_keys.get("etherscan"))
    forecast_result = predictive_forecasting(project, api_key=api_keys.get("forecast"))
    risk_rating = calculate_risk_rating(
        sentiment_score=sentiment_result["score"],
        technical=technical_result,
        onchain=onchain_result,
        forecast=forecast_result
    )
    recommendations = generate_recommendations(risk_rating)
    report = RiskAssessmentReport(
        project=project,
        sentiment_score=sentiment_result["score"],
        sentiment_summary=sentiment_result["summary"],
        technical_indicators=technical_result,
        onchain_metrics=onchain_result,
        forecast=forecast_result,
        risk_rating=risk_rating,
        recommendations=recommendations
    )
    return report.json(), "Assessment completed successfully.", {"project": project, "status": "completed"}, None


# --- ENTRY POINT --- #

ALLOWED_TOOLS = {
    "cert_risk": run_cert_assessment,
}


@with_key_rotation
def run(**kwargs) -> Tuple[str, Optional[str], Optional[Dict[str, Any]], Any, Any]:
    """
    Main entry point for the CERT risk assessment mech.
    
    Expects kwargs to contain:
      - tool: should be "cert_risk"
      - prompt: a JSON string with "project" and "contract_address" keys
      - api_keys: a dict containing required API keys (e.g., "openai", "taapi", "etherscan", "forecast")
    
    Returns a MechResponse tuple.
    """
    tool: Optional[str] = kwargs.get("tool", None)
    if tool is None:
        return error_response("No tool has been specified.") + (None,)
    
    if tool not in ALLOWED_TOOLS:
        return error_response(f"Tool {tool!r} is not supported. Supported tools: {tuple(ALLOWED_TOOLS.keys())}.") + (None,)
    
    prompt: Optional[str] = kwargs.get("prompt", None)
    if prompt is None:
        return error_response("No prompt has been given.") + (None,)
    
    try:
        params = json.loads(prompt)
        project = params.get("project")
        contract_address = params.get("contract_address")
        if not project or not contract_address:
            return error_response("Both 'project' and 'contract_address' must be provided in the prompt.") + (None,)
    except Exception as e:
        return error_response(f"Invalid prompt format: {e}") + (None,)
    
    api_keys = kwargs.get("api_keys", {})
    if not api_keys.get("openai"):
        return error_response("No OpenAI API key has been provided.") + (None,)
    if not api_keys.get("etherscan"):
        return error_response("No Etherscan API key has been provided.") + (None,)
    
    # Execute the CERT risk assessment.
    transaction_builder = ALLOWED_TOOLS[tool]
    return transaction_builder(project, contract_address, api_keys)
