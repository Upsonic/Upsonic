# Lazy import cache for heavy dependencies
_FASTAPI_IMPORTS = None


def get_fastapi_imports():
    """Lazy load FastAPI dependencies only when needed."""
    global _FASTAPI_IMPORTS
    if _FASTAPI_IMPORTS is None:
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            import uvicorn
            
            _FASTAPI_IMPORTS = {
                'FastAPI': FastAPI,
                'Request': Request,
                'JSONResponse': JSONResponse,
                'uvicorn': uvicorn,
            }
        except ImportError:
            return None
    return _FASTAPI_IMPORTS

