import os

def ensure_directory(path):
    """Create directory if it doesn't exist, preserving symlinks."""
    if os.path.islink(path):
        target = os.readlink(path)
        if not os.path.exists(target):
            print(f"Creating target directory for symlink {path} -> {target}")
            os.makedirs(target, exist_ok=True)
    elif not os.path.exists(path):
        print(f"Creating directory: {path}")
        os.makedirs(path, exist_ok=True)
    else:
        print(f"Directory already exists: {path}")

if __name__ == "__main__":
    # Check and create weight directories
    ensure_directory("/runpod-volume/assets/weights")
    ensure_directory("/runpod-volume/opt")
    print("Directory check completed")
