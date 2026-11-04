"""
@Author: Srini Yedluri
@Date: 3/27/26
@Time: 12:29 PM
@File: main.py
"""

"""
main.py — start the FastAPI app on port 8003.
in case if we need to run but for application this the 
entry point.
"""
import uvicorn
import logging

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
