from pathlib import Path

# Root project name
PROJECT_NAME = "fintech-risk-intelligence"

# Folder structure
folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models",
    "reports",
    "api",
    "app",
    "tests",
    "docker"
]

# Files to create
files = [
    ".env",
    "README.md",
    ".gitignore",
    "api/main.py",
    "app/streamlit_app.py",
    "tests/test_basic.py"
]


def create_structure():
    for folder in folders:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created folder: {path}")

    for file in files:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        print(f"Created file: {path}")


if __name__ == "__main__":
    create_structure()
    print("\nProject structure created successfully 🚀")