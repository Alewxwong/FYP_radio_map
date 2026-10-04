import os
import csv
import numpy as np
import h5py
import multiprocessing as mp
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.draw import line
from tqdm import tqdm

# ==========================================================
# 1. CONFIGURATION AND PATHS
# ==========================================================
BASE_DIR = r"C:\Users\user\Desktop\Fgo\dataset\RadioMapSeer"
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Helper to handle accidental trailing spaces in folder names
def choose_dir(base, names):
    for name in names:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    return os.path.join(base, names[0])

PNG_DIR = choose_dir(BASE_DIR, ["png", "png "])
BUILDINGS_DIR = choose_dir(PNG_DIR, ["buildings_complete", "buildings_complete "])
ANTENNAS_DIR = choose_dir(PNG_DIR, ["antennas", "antennas "])

GAIN_DIR = choose_dir(BASE_DIR, ["gain", "gain "])
IRT2_DIR = choose_dir(GAIN_DIR, ["IRT2", "IRT2 "])
IRT4_DIR = choose_dir(GAIN_DIR, ["IRT4", "IRT4 "])

# Output directory (kept as the original name)
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

IMG_SIZE = 256
NUM_INPUT_CHANNELS = 5  # Building, Antenna, SDF, Tx_Dist, LoS
NUM_TRANSMITTERS = 80
BATCH_SIZE = 200        # Process and save 200 samples at a time to save RAM

NUM_WORKERS = max(1, mp.cpu_count() - 1)  # Use all CPU cores except 1

try:
    BILINEAR = Image.Resampling.BILINEAR
except Exception:
    BILINEAR = Image.BILINEAR


# ==========================================================
# 2. HELPER FUNCTIONS
# ==========================================================
def load_image(path):
    img = Image.open(path).convert("L")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), BILINEAR)
    return np.asarray(img, dtype=np.float32)

def get_tx_coordinates_from_mask(antenna_mask):
    coords = np.argwhere(antenna_mask > 0.5)
    if len(coords) > 0:
        tx_y = int(np.mean(coords[:, 0]))
        tx_x = int(np.mean(coords[:, 1]))
        return tx_x, tx_y
    return IMG_SIZE // 2, IMG_SIZE // 2

def compute_sdf(building_mask):
    """
    True Signed Distance Field, remapped to [0, 1].
    0.0 = deep inside building
    0.5 = building boundary
    1.0 = far outside building
    """
    building_mask = (building_mask > 0.5).astype(np.float32)
    max_dist = float(np.sqrt(2.0) * max(1, IMG_SIZE - 1))
    
    inside = distance_transform_edt(building_mask)
    outside = distance_transform_edt(1.0 - building_mask)
    
    sdf = outside - inside
    sdf = np.clip(sdf / max_dist, -1.0, 1.0)
    
    # Remap [-1, 1] -> [0, 1]
    sdf = (sdf + 1.0) / 2.0
    return sdf.astype(np.float32)

def compute_tx_distance_map(tx_x, tx_y):
    y_coords, x_coords = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist_map = np.sqrt((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2)
    max_dist = float(np.sqrt(2.0) * max(1, IMG_SIZE - 1))
    return (dist_map / max_dist).astype(np.float32)

def compute_antenna_heatmap(tx_x, tx_y):
    y_coords, x_coords = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    sigma = 3.0
    gaussian = np.exp(-((x_coords - tx_x) ** 2 + (y_coords - tx_y) ** 2) / (2.0 * sigma ** 2))
    max_val = gaussian.max()
    if max_val < 1e-8:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    return (gaussian / max_val).astype(np.float32)

def compute_los_mask_exact(building_mask, tx_x, tx_y):
    """
    Exact pixel-by-pixel LoS mask. (Computationally heavy but accurate).
    """
    bool_mask = building_mask > 0.5
    los_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    if bool_mask[tx_y, tx_x]:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            if x == tx_x and y == tx_y:
                continue
            rr, cc = line(tx_y, tx_x, y, x)
            if len(rr) > 2:
                if np.any(bool_mask[rr[1:-1], cc[1:-1]]):
                    los_mask[y, x] = 0.0
    return los_mask


# ==========================================================
# 3. CORE PROCESSING FUNCTION
# ==========================================================
def process_single_sample(args):
    map_id, tx_id, building_path, antenna_path, irt2_path, irt4_path = args

    building_raw = load_image(building_path) / 255.0
    antenna_mask = load_image(antenna_path) / 255.0
    irt2_map = load_image(irt2_path)

    building_mask = (building_raw > 0.5).astype(np.float32)
    tx_x, tx_y = get_tx_coordinates_from_mask(antenna_mask)

    sdf = compute_sdf(building_mask)
    tx_dist = compute_tx_distance_map(tx_x, tx_y)
    ant_heatmap = compute_antenna_heatmap(tx_x, tx_y)
    los_mask = compute_los_mask_exact(building_mask, tx_x, tx_y)

    input_channels = np.stack([building_mask, ant_heatmap, sdf, tx_dist, los_mask], axis=0).astype(np.float32)
    
    # Safety clip to [0, 1]
    input_channels = np.clip(input_channels, 0.0, 1.0)

    if os.path.exists(irt4_path):
        irt4_map = load_image(irt4_path)
    else:
        irt4_map = np.full((IMG_SIZE, IMG_SIZE), -1.0, dtype=np.float32)

    return {
        'map_id': int(map_id),
        'tx_id': int(tx_id),
        'inputs': input_channels,
        'irt2_target': irt2_map[np.newaxis, ...].astype(np.float32),
        'irt4_target': irt4_map[np.newaxis, ...].astype(np.float32)
    }


# ==========================================================
# 4. PARSE CSV
# ==========================================================
def parse_samples():
    print("Parsing dataset.csv...")
    sample_list = []
    skipped = 0

    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_name = row.get("maps", None)
            if not map_name:
                skipped += 1
                continue
            
            map_id_str = os.path.splitext(map_name)[0]
            try:
                map_id = int(map_id_str)
            except Exception:
                skipped += 1
                continue

            building_path = os.path.join(BUILDINGS_DIR, map_name)
            if not os.path.exists(building_path):
                skipped += 1
                continue

            for tx_idx in range(NUM_TRANSMITTERS):
                gain_col_name = f"Gain{tx_idx + 1}"
                gain_name = row.get(gain_col_name, None)
                if not gain_name:
                    continue

                antenna_path = os.path.join(ANTENNAS_DIR, gain_name)
                irt2_path = os.path.join(IRT2_DIR, gain_name)
                irt4_path = os.path.join(IRT4_DIR, gain_name)

                if not os.path.exists(antenna_path) or not os.path.exists(irt2_path):
                    skipped += 1
                    continue

                sample_list.append({
                    'map_id': map_id, 'tx_id': tx_idx,
                    'building_path': building_path, 'antenna_path': antenna_path,
                    'irt2_path': irt2_path, 'irt4_path': irt4_path
                })

    print(f"Total valid samples found: {len(sample_list)}")
    print(f"Skipped / missing entries: {skipped}")
    return sample_list


# ==========================================================
# 5. MAIN EXECUTION
# ==========================================================
def main():
    print("Starting RadioMapSeer Data Processing (Full Generation)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    sample_list = parse_samples()

    train_samples = [s for s in sample_list if int(s['map_id']) < 560]
    val_samples = [s for s in sample_list if 560 <= int(s['map_id']) < 630]
    test_samples = [s for s in sample_list if int(s['map_id']) >= 630]

    splits = {
        'train': train_samples, 
        'val': val_samples, 
        'test': test_samples
    }

    print("\nSplit sizes:")
    for name in ["train", "val", "test"]:
        print(f"  {name}: {len(splits[name])}")

    for split_name, samples in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split ({len(samples)} samples)")
        print(f"{'='*60}")

        if len(samples) == 0:
            continue

        output_file = os.path.join(OUTPUT_DIR, f"radiomapseer_multifidelity_{split_name}.h5")

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('inputs', shape=(len(samples), NUM_INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype='float32')
            f.create_dataset('irt2_targets', shape=(len(samples), 1, IMG_SIZE, IMG_SIZE), dtype='float32')
            f.create_dataset('irt4_targets', shape=(len(samples), 1, IMG_SIZE, IMG_SIZE), dtype='float32')
            f.create_dataset('map_ids', shape=(len(samples),), dtype='int32')
            f.create_dataset('tx_ids', shape=(len(samples),), dtype='int32')

            tasks = [
                (s['map_id'], s['tx_id'], s['building_path'], s['antenna_path'], s['irt2_path'], s['irt4_path']) 
                for s in samples
            ]

            # Multiprocessing pool for CPU-heavy LoS and SDF calculations
            with mp.Pool(processes=NUM_WORKERS) as pool:
                for batch_start in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Writing {split_name}"):
                    batch_end = min(batch_start + BATCH_SIZE, len(samples))
                    batch_tasks = tasks[batch_start:batch_end]
                    
                    # Process batch in parallel
                    batch_results = list(pool.imap(process_single_sample, batch_tasks))
                    
                    # Write batch directly to disk (Main process handles H5 writing)
                    for i, res in enumerate(batch_results):
                        idx = batch_start + i
                        f['inputs'][idx] = res['inputs']
                        f['irt2_targets'][idx] = res['irt2_target']
                        f['irt4_targets'][idx] = res['irt4_target']
                        f['map_ids'][idx] = res['map_id']
                        f['tx_ids'][idx] = res['tx_id']

        print(f"Successfully saved {split_name} split to {output_file}!")

    print("\n" + "="*60)
    print("Data processing complete!")
    print("="*60)
    print(f"\nYour new dataset is saved in:\n{OUTPUT_DIR}")
    
if __name__ == "__main__":
    mp.freeze_support()  # Required for Windows multiprocessing
    main()