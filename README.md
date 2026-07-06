# temporal_serial_dictator

## Installation

Requires Python 3.11+. Check your version with `python3 --version`.

### Option A: with Task (recommended)

[Task](https://taskfile.dev/installation/) is a small tool that runs the commands below for you.

1. Install Task (see link above for your OS).
2. From the repo root, run:

   ```
   task install
   ```

This creates a virtual environment in `.venv` and installs the project into it.

### Option B: without Task

1. Create a virtual environment:

   ```
   python3 -m venv .venv
   ```

2. Activate it:

   ```
   source .venv/bin/activate
   ```

   (On Windows: `.venv\Scripts\activate`. See the [venv docs](https://docs.python.org/3/library/venv.html) for details.)

3. Install the project:

   ```
   pip install -e '.[dev]'
   ```

You'll need to run the `source .venv/bin/activate` step again each time you open a new terminal.