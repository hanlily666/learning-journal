"""Write the API's OpenAPI schema to server/openapi.json (the source of the browser's types).

    uv run python -m scripts.export_openapi && (cd ../web && npm run gen:api)
"""

import json
from pathlib import Path

from app.main import create_app

out = Path(__file__).resolve().parent.parent / "openapi.json"
out.write_text(json.dumps(create_app().openapi(), indent=2) + "\n")
print(f"wrote {out}")
