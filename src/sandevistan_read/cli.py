from __future__ import annotations

import uvicorn

from .config import CONFIG
from .paths import PATHS


def main() -> None:
    PATHS.ensure()
    uvicorn.run("sandevistan_read.app:app", host=CONFIG.server.host, port=CONFIG.server.port, log_level=CONFIG.server.log_level)


if __name__ == "__main__": main()
