# utilities to manage environment, allowing to run the notebook either using google colab or in a local runtime
from pathlib import Path

def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def project_root():
    current = Path.cwd().resolve()

    for path in [current] + list(current.parents):
        if (path / ".git").exists():
            return path

    raise RuntimeError("Project root not found")
