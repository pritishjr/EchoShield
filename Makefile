run:
	uv run uvicorn app.main:app --reload

test:
	uv run pytest

profile:
	uv run python scripts/profile_pipeline.py