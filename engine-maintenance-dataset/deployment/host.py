from huggingface_hub import HfApi

api = HfApi()

space_repo = "ashuPandey/vehicle_predictive_maintenance"

api.upload_folder(
    folder_path="vehicle_predictive_maintenance/deployment",
    repo_id=space_repo,
    repo_type="space"
)

print("Deployment files pushed to Hugging Face Space successfully.")
