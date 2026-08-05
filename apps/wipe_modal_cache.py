import os
import shutil
import modal

app = modal.App("wipe-latin-itn-cache")
cache_volume = modal.Volume.from_name("latin-itn-cache", create_if_missing=True)


@app.function(volumes={"/mnt/cache": cache_volume})
def wipe():
    target_dir = "/mnt/cache"

    if os.path.exists(target_dir):
        for item in os.listdir(target_dir):
            item_path = os.path.join(target_dir, item)
            try:
                if os.path.isdir(item_path) and not os.path.islink(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            except Exception as e:
                print(f"Could not remove {item_path}: {e}")

        print(f"Successfully wiped contents of {target_dir}!")

    # Persist the deletions to the Modal Volume
    cache_volume.commit()


@app.local_entrypoint()
def main():
    wipe.remote()