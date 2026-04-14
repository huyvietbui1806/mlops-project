import os
import gdown

# ===== 1. Define output directory =====
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 2. Google Drive file IDs =====
TRAIN_FILE_ID = "18PQSky51zCjLbX-wKJ8Meb9ZFe4Pxxjq"
TEST_FILE_ID  = "1bpOkO9WV6saLWMxF4swk1Ukiw9b_s3oP"  

# ===== 3. Output file paths =====
train_output_path = os.path.join(OUTPUT_DIR, "train.csv")
test_output_path  = os.path.join(OUTPUT_DIR, "test.csv")

# ===== 4. Download URLs =====
train_url = f"https://drive.google.com/uc?id={TRAIN_FILE_ID}"
test_url  = f"https://drive.google.com/uc?id={TEST_FILE_ID}"

# ===== 5. Download function =====
def download_file(url, output_path):
    print(f"Downloading to {output_path}...")
    gdown.download(url, output_path, quiet=False)
    print("Done.\n")

# ===== 6. Run =====
if __name__ == "__main__":
    download_file(train_url, train_output_path)
    download_file(test_url, test_output_path)

    print("All files downloaded successfully!")