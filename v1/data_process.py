import os
import csv
import numpy as np
import h5py
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.draw import line
from tqdm import tqdm
import multiprocessing as mp

# ==========================================
# 1. CONFIGURATION AND PATHS
# ==========================================
BASE_DIR = r"C:\Users\user\Desktop\Fgo\dataset\RadioMapSeer"
CSV_PATH = os.path.join(BASE_DIR, "dataset.csv")

BUILDINGS_DIR = os.path.join(BASE_DIR, "png", "buildings_complete")
ANTENNAS_DIR = os.path.join(BASE_DIR, "png", "antennas")
IRT2_DIR = os.path.join(BASE_DIR, "gain", "IRT2")
IRT4_DIR = os.path.join(BASE_DIR, "gain", "IRT4")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

IMG_SIZE = 256 
NUM_INPUT_CHANNELS = 5 # Building, Antenna, SDF, Tx_Dist, LoS
BATCH_SIZE = 500       # Process and save 500 samples at a time to save RAM

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_tx_coordinates_from_mask(antenna_mask):
    coords = np.argwhere(antenna_mask > 0.5)
    if len(coords) > 0:
        tx_y = int(np.mean(coords[:, 0]))
        tx_x = int(np.mean(coords[:, 1]))
        return tx_x, tx_y
    return IMG_SIZE // 2, IMG_SIZE // 2

def compute_sdf(building_mask):
    sdf = distance_transform_edt(1 - building_mask)
    max_dist = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
    return (sdf / max_dist).astype(np.float32)

def compute_tx_distance_map(tx_x, tx_y):
    y_coords, x_coords = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    dist_map = np.sqrt((x_coords - tx_x)**2 + (y_coords - tx_y)**2)
    max_dist = np.sqrt(IMG_SIZE**2 + IMG_SIZE**2)
    return (dist_map / max_dist).astype(np.float32)

def compute_los_mask(building_mask, tx_x, tx_y):
    los_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for y in range(IMG_SIZE):
        for x in range(IMG_SIZE):
            if x == tx_x and y == tx_y: continue
            rr, cc = line(tx_y, tx_x, y, x)
            if len(rr) > 2:
                path_pixels = building_mask[rr[1:-1], cc[1:-1]]
                if np.any(path_pixels > 0.5):
                    los_mask[y, x] = 0.0
    return los_mask

def compute_antenna_heatmap(tx_x, tx_y):
    y_coords, x_coords = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    sigma = 3.0
    gaussian = np.exp(-((x_coords - tx_x)**2 + (y_coords - tx_y)**2) / (2 * sigma**2))
    return (gaussian / gaussian.max()).astype(np.float32)

# ==========================================
# 3. CORE PROCESSING FUNCTION
# ==========================================
def process_single_sample(args):
    map_id, tx_id, building_path, antenna_path, irt2_path, irt4_path = args
    
    building_mask = np.array(Image.open(building_path).convert('L')) / 255.0
    antenna_mask = np.array(Image.open(antenna_path).convert('L')) / 255.0
    irt2_map = np.array(Image.open(irt2_path).convert('L')).astype(np.float32)
    
    building_mask = (building_mask > 0.5).astype(np.float32)
    tx_x, tx_y = get_tx_coordinates_from_mask(antenna_mask)
    
    sdf = compute_sdf(building_mask)
    tx_dist = compute_tx_distance_map(tx_x, tx_y)
    los_mask = compute_los_mask(building_mask, tx_x, tx_y)
    ant_heatmap = compute_antenna_heatmap(tx_x, tx_y)
    
    # Stack 5 channels
    input_channels = np.stack([building_mask, ant_heatmap, sdf, tx_dist, los_mask], axis=0)
    
    if os.path.exists(irt4_path):
        irt4_map = np.array(Image.open(irt4_path).convert('L')).astype(np.float32)
    else:
        irt4_map = np.full((IMG_SIZE, IMG_SIZE), -1.0, dtype=np.float32)
        
    return {
        'map_id': int(map_id),
        'tx_id': int(tx_id),
        'inputs': input_channels,
        'irt2_target': irt2_map[np.newaxis, ...],
        'irt4_target': irt4_map[np.newaxis, ...]
    }

# ==========================================
# 4. MAIN EXECUTION (Memory Efficient)
# ==========================================
def main():
    print("Starting Multi-Fidelity Data Processing...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Parsing dataset.csv...")
    sample_list = []
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            map_name = row['maps']
            map_id = map_name.split('.')[0]
            building_path = os.path.join(BUILDINGS_DIR, map_name)
            
            for tx_idx in range(80):
                gain_col_name = f"Gain{tx_idx + 1}"
                gain_name = row[gain_col_name]
                
                antenna_path = os.path.join(ANTENNAS_DIR, gain_name)
                irt2_path = os.path.join(IRT2_DIR, gain_name)
                irt4_path = os.path.join(IRT4_DIR, gain_name)
                
                sample_list.append({
                    'map_id': map_id, 'tx_id': tx_idx,
                    'building_path': building_path, 'antenna_path': antenna_path,
                    'irt2_path': irt2_path, 'irt4_path': irt4_path
                })
                
    print(f"Total samples found: {len(sample_list)}")
    
    # Split data by Map ID
    train_samples = [s for s in sample_list if int(s['map_id']) < 560]
    val_samples = [s for s in sample_list if 560 <= int(s['map_id']) < 630]
    test_samples = [s for s in sample_list if int(s['map_id']) >= 630]
    
    #splits = {'train': train_samples, 'val': val_samples, 'test': test_samples}
    splits = {'test': test_samples}
    num_workers = max(1, mp.cpu_count() - 1)
    
    for split_name, samples in splits.items():
        print(f"\n--- Processing {split_name.upper()} split ({len(samples)} samples) ---")
        output_file = os.path.join(OUTPUT_DIR, f"radiomapseer_multifidelity_{split_name}.h5")
        
        # Pre-allocate the HDF5 file structure on disk
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('inputs', shape=(len(samples), NUM_INPUT_CHANNELS, IMG_SIZE, IMG_SIZE), dtype='float32')
            f.create_dataset('irt2_targets', shape=(len(samples), 1, IMG_SIZE, IMG_SIZE), dtype='float32')
            f.create_dataset('irt4_targets', shape=(len(samples), 1, IMG_SIZE, IMG_SIZE), dtype='float32')
            f.create_dataset('map_ids', shape=(len(samples),), dtype='int32')
            f.create_dataset('tx_ids', shape=(len(samples),), dtype='int32')
            
            # Process and write in batches to avoid MemoryError
            for batch_start in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Writing {split_name}"):
                batch_end = min(batch_start + BATCH_SIZE, len(samples))
                batch_samples = samples[batch_start:batch_end]
                
                tasks = [
                    (s['map_id'], s['tx_id'], s['building_path'], s['antenna_path'], 
                     s['irt2_path'], s['irt4_path']) for s in batch_samples
                ]
                
                # Process batch
                with mp.Pool(processes=num_workers) as pool:
                    batch_results = list(pool.imap(process_single_sample, tasks))
                
                # Write batch directly to disk
                for i, res in enumerate(batch_results):
                    idx = batch_start + i
                    f['inputs'][idx] = res['inputs']
                    f['irt2_targets'][idx] = res['irt2_target']
                    f['irt4_targets'][idx] = res['irt4_target']
                    f['map_ids'][idx] = res['map_id']
                    f['tx_ids'][idx] = res['tx_id']
                    
        print(f"Successfully saved {split_name} split to {output_file}!")

if __name__ == "__main__":
    main()