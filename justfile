set positional-arguments

# List available commands.
default:
    @just --list

# Install dependencies and check the environment.
setup: sync check

# Install project dependencies from the lockfile.
sync:
    mise exec -- uv sync --locked

# Check dependencies and the selected device without downloading model weights.
check *args:
    mise exec -- uv run main.py --check "$@"

# Check Python lint and formatting.
lint:
    mise exec -- uv run ruff check .
    mise exec -- uv run ruff format --check .

# Fix lint issues and format Python code.
format:
    mise exec -- uv run ruff check --fix .
    mise exec -- uv run ruff format .

# Run inference with optional command-line arguments.
run *args:
    mise exec -- uv run main.py "$@"

# Compare the login sample with Harbor; reads TYPESAFE_API_KEY from .env.
eval *args:
    PYTHONPATH=. mise exec -- uv run --locked --group eval --env-file .env harbor run -c eval.yaml "$@"

# Open the Harbor results viewer.
eval-report *args:
    mise exec -- uv run --locked --group eval harbor view jobs "$@"
