"""Run the FastAPI app standalone (``python src/main.py``) with uvicorn on port 8007.

The Docker image starts uvicorn directly on ``api.app:app`` (see the Dockerfile CMD).
"""
import logging

import uvicorn

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info(" Application starting in standalone mode")

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8007,
        reload=False,
        log_level="info",
    )
