import os
import json
import huggingface_hub


class HuggingFaceModelManager:
    """
    Model manager for downloading and verifying models from Hugging Face Hub.
    """

    def __init__(self):
        hf_token = os.environ.get("HF_TOKEN", None)
        if hf_token is not None:
            huggingface_hub.login(token=hf_token)
            print("Logged in to Hugging Face Hub successfully")
        else:
            print(
                "Warning: HF_TOKEN not found in environment variables. Anonymous access will be used, which may have rate limits."
            )

    def verify_config(self, config):
        """
        Verify that the model config meets the required format.
        """
        # Check existence of required fields
        if "arch_type" not in config:
            return {"statusCode": 400, "body": "arch_type not found"}

        if "arch_version" not in config:
            return {"statusCode": 400, "body": "arch_version not found"}

        if "components" not in config:
            return {"statusCode": 400, "body": "components not found"}

        # Check field types
        if not isinstance(config["arch_type"], str):
            return {"statusCode": 400, "body": "arch_type must be str"}

        if not isinstance(config["arch_version"], str):
            return {"statusCode": 400, "body": "arch_version must be str"}

        if not isinstance(config["components"], dict):
            return {"statusCode": 400, "body": "components must be dict"}

        # Validate architecture type
        if config["arch_type"] != "rvc":
            return {"statusCode": 400, "body": "arch_type must be 'rvc'"}

        # Check required components
        if "pth" not in config["components"]:
            return {"statusCode": 400, "body": "components['pth'] not found"}

        if "index" not in config["components"]:
            return {"statusCode": 400, "body": "components['index'] not found"}

        return None

    def get_model(self, model_name):
        """
        Download a model and its components from Hugging Face Hub.

        Args:
            model_name (str): The Hugging Face repository ID (e.g., username/model-name)

        Returns:
            dict: Response with status code and body containing model information
        """
        try:
            print(f"Downloading config for model: {model_name}")
            config_path = huggingface_hub.hf_hub_download(
                repo_id=model_name, filename="config.json", repo_type="model"
            )

            with open(config_path, "r") as f:
                config = json.load(f)

            check = self.verify_config(config)
            if check is not None:
                return check

            print(f"Downloading model file: {config['components']['pth']}")
            pth_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename=config["components"]["pth"],
                repo_type="model",
            )

            print(f"Downloading index file: {config['components']['index']}")
            index_path = huggingface_hub.hf_hub_download(
                repo_id=model_name,
                filename=config["components"]["index"],
                repo_type="model",
            )

            print("Model download complete")
            return {
                "statusCode": 200,
                "body": {
                    "config": config,
                    "pth_path": pth_path,
                    "index_path": index_path,
                },
            }
        except Exception as e:
            error_message = f"Error downloading model: {str(e)}"
            print(error_message)
            return {"statusCode": 400, "body": error_message}
