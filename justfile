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
