import uvicorn
from .app_factory import create_app
from .config import load_config_from_env


config = load_config_from_env()
app = create_app(config)

if __name__ == "__main__":
    uvicorn.run(app, host=config.host, port=config.port)
