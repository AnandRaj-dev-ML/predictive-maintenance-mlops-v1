from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="predictive_maintenance_mlops_v1/deployment",     # the local folder containing your files
    repo_id="Rajanan/predictive-engine-maintenance-v1",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
