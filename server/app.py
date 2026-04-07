"""
OpenEnv spec standard app entry point wrapper.
"""
from api import app

def main():
    import uvicorn
    import os
    uvicorn.run("server.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), workers=1)

if __name__ == "__main__":
    main()
