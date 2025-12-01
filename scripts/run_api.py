"""Start the FastAPI server using configuration from .env."""

import uvicorn

from src.utils.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

