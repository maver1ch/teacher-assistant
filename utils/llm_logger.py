# utils/llm_logger.py
import logging
import os
from typing import Any

# Import pricing from config
from utils.config import MODEL_PRICING

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "llm_calls.log")
LLM_LOGGER_NAME = "llm_logger"

def setup_llm_logger():
    """
    Sets up a dedicated logger for LLM API calls.
    Creates the log directory and file handler.
    """
    logger = logging.getLogger(LLM_LOGGER_NAME)
    
    # Prevent adding duplicate handlers
    if logger.hasHandlers():
        return

    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Create a file handler to write logs to a file
    handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    
    # Define the log format
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)

def _calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculates the cost of an API call based on the model and token usage.
    """
    pricing = MODEL_PRICING.get(model_name)
    if not pricing:
        return 0.0
        
    try:
        # Price is per 1 million tokens
        input_cost = (prompt_tokens / 1_000_000) * pricing.get("input", 0.0)
        output_cost = (completion_tokens / 1_000_000) * pricing.get("output", 0.0)
        return input_cost + output_cost
    except Exception:
        return 0.0

def log_llm_call(response: Any, model_name: str, service_name: str):
    """
    Logs the details of an OpenAI API call, including token usage and estimated cost.

    Args:
        response: The response object from the OpenAI client.
        model_name: The name of the model that was called.
        service_name: A string identifying the service making the call (e.g., 'exam_analysis').
    """
    # Ensure logger is set up
    setup_llm_logger()
    logger = logging.getLogger(LLM_LOGGER_NAME)
    
    try:
        if not response or not hasattr(response, 'usage') or not response.usage:
            logger.warning(f"[LLM Call] Service: {service_name} | Model: {model_name} | Usage data not available in response.")
            return

        usage = response.usage
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens or 0
        
        # Calculate cost
        cost = _calculate_cost(model_name, prompt_tokens, completion_tokens)
        
        # Get a snippet of the response content
        snippet = ""
        if response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""
            snippet = content.replace('\n', ' ').strip()
            snippet = snippet
            
        log_message = (
            f"[LLM Call] Service: {service_name} | Model: {model_name} | "
            f"Tokens: {total_tokens} (P: {prompt_tokens}, C: {completion_tokens}) | "
            f"Cost: ${cost:.6f} | "
            f"Snippet: \"{snippet}\""
        )
        
        logger.info(log_message)

    except Exception as e:
        logger.error(f"Failed to log LLM call for service {service_name}: {e}")