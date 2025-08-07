from pathlib import Path
import numpy as np
from tqdm import tqdm
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json

from utils import find_contour, resample_contour

if __name__ == "__main__":
    # Configuration
    img_size = 128 
    contour_length = 128
    num_samples_per_category = 1000
    base_dir = Path("original_data/cell_segmentations")
    output_path = Path("generated_data")
    output_file = output_path / "cell_segmentations.parquet"
    output_file_metadata = output_path / "cell_segmentations_metadata.json"
    seed = 0
    rng = np.random.default_rng(seed)
    # Create dataset
    output_path.mkdir(exist_ok=True)
    rows = []
    for dir in sorted(base_dir.iterdir()):
        if not dir.is_dir():
            continue
        print(f"Processing directory: {dir.name}")
        values = dir.name.split("_")
        ecc = float(".".join(values[1].split("-")[1:]))
        rand = float(".".join(values[2].split("-")[1:]))
        print(f"Parameters: {ecc=}, {rand=}")
        files_dir = dir / "data/processed"
        filenames = list(files_dir.glob("*.png"))
        # Shuffle the filenames to ensure randomness
        rng.shuffle(filenames)
        filenames = filenames[:num_samples_per_category]
        print(f"Found {len(filenames)} files.")

        for filename in tqdm(filenames, desc=f"Processing {dir.name}"):
            image = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            contour = find_contour(image)
            image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

            contour_resized = find_contour(image)
            #contour_image = np.zeros_like(image)
            #cv2.drawContours(contour_image, [contour_resized], -1, 255, 1)
            
            filled_image = np.zeros_like(image)
            cv2.drawContours(filled_image, [contour_resized], -1, 255, -1)

            contour = resample_contour(contour, contour_length)
            contour -= np.mean(contour, axis=0)  # Center the contour

            row = {
                "ecc": ecc,
                "rand": rand,
                #"contour_image": contour_image.reshape(-1) / 255.0,
                "filled_image": filled_image.reshape(-1) / 255.0,
                "contour": contour.reshape(-1),
            }

            rows.append(row)

    print(f"Processed {len(rows)} rows.")
    print("Saving dataset to Parquet format...")   
    df = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_file)
    print(f"Saved dataset to {output_file}")

    sample_size = min(5000, len(rows))
    sampled_rows = rng.choice(rows, size=sample_size, replace=False)
    #contour_images = np.array([row['contour_image'] for row in sampled_rows])
    filled_images = np.array([row['filled_image'] for row in sampled_rows])
    #mean_contour = np.mean(contour_images)
    #std_contour = np.std(contour_images)
    mean_filled = np.mean(filled_images)
    std_filled = np.std(filled_images)

    #print(f"Mean contour image: {mean_contour}")
    #print(f"Std contour image: {std_contour}")
    print(f"Mean filled image: {mean_filled}")
    print(f"Std filled image: {std_filled}")

    metadata = {
        "label": "PCST Cell Segmentation Contours",
        "description": "Contour dataset based on the PCST dataset.",
        "image_width": img_size,
        "image_height": img_size,
        "contour_length": contour_length,
        #"mean_contour_image": mean_contour,
        #"std_contour_image": std_contour,
        "mean_filled_image": mean_filled,
        "std_filled_image": std_filled,
    }

    output_file_metadata.parent.mkdir(exist_ok=True, parents=True)
    with open(output_file_metadata, "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {output_file_metadata}")

