import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess TSMC2014 Raw Data to STAN .npy format")
    parser.add_argument('input_file', type=str, help='Path to raw .txt/.tsv file')
    parser.add_argument('output_dir', type=str, help='Directory to save output .npy files')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset (e.g., NYC)')
    parser.add_argument('--min_checkins', type=int, default=10, help='Filter users with fewer than N checkins')
    return parser.parse_args()

def preprocess(args):
    print(f"Loading raw data from {args.input_file}...")
    # Columns based on user description:
    # 1) user_id 2) venue_id 3) venue_category_id 4) venue_category_name 5) latitude 6) longitude 7) timezone 8) utc_time
    # TSV format
    # Note: Some lines might be malformed, so we use error_bad_lines=False or on_bad_lines='skip' depending on pandas version
    try:
        df = pd.read_csv(args.input_file, sep='\t', header=None, encoding='latin-1', 
                         names=['user_original', 'venue_original', 'cat_id', 'cat_name', 'lat', 'lon', 'tz', 'utc_time'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Raw data shape: {df.shape}")

    # 1. Filter Users
    user_counts = df['user_original'].value_counts()
    valid_users = user_counts[user_counts >= args.min_checkins].index
    df = df[df['user_original'].isin(valid_users)].copy()
    print(f"Filtered data shape (min_checkins={args.min_checkins}): {df.shape}")

    # 2. ID Mapping
    # Users
    user_mapping = {u: i+1 for i, u in enumerate(df['user_original'].unique())} # 1-based
    df['user_id'] = df['user_original'].map(user_mapping)
    
    # POIs (Venues)
    # Different venues might have same lat/lon? The paper says POI. 
    # Usually we map by VenueID.
    venue_mapping = {v: i+1 for i, v in enumerate(df['venue_original'].unique())} # 1-based
    df['loc_id'] = df['venue_original'].map(venue_mapping)

    print(f"Number of Users: {len(user_mapping)}")
    print(f"Number of POIs: {len(venue_mapping)}")

    # 3. Time Conversion
    # Format: Mon Apr 04 18:28:57 +0000 2013 or similar?
    # User said: "2012-04-12 到 2013-02-16"
    # Example TSMC line usually looks like: Tue Apr 03 18:00:09 +0000 2012
    # We need to parse this.
    
    # Let's try to infer format or use a standard parser
    # Using pandas to_datetime is easiest, but might be slow.
    print("Parsing timestamps...")
    df['dt'] = pd.to_datetime(df['utc_time'], format='%a %b %d %H:%M:%S +0000 %Y')
    
    # Convert to minutes relative to earliest time in dataset (or absolute epoch minutes? STAN load.py uses /60)
    # Existing load.py: data[:, -1] = np.array(data[:, -1]/60, dtype=int)
    # This implies the input npy time column is in Seconds? Or already minutes?
    # Wait, load.py line 68: `data[:, -1] = np.array(data[:, -1]/60, dtype=int)`
    # This suggests the .npy file usually contains seconds (if dividing by 60 gives minutes) or minutes (if dividing by 60 gives hours).
    # MultiEmbed uses: `traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1` where `hours = 24*7`.
    # This implies the time step in the model is "Hours".
    # So `process_traj` produces "Hours" in `traj`.
    # Let's look at `process_traj` again.
    # `user_traj[:, 2]` is the time column.
    # `data[:, -1] = data[:, -1]/60`. If input is Seconds, this becomes Minutes.
    # But later `traj[:, :, 2] % (24*7)` implies Hours.
    # If `process_traj` does `/60`, and the result is used for `24*7` modulo, then the result of `/60` must be Hours.
    # So the input to `load.py` (the .npy file) must be in **Minutes**.  (Minutes / 60 = Hours).
    # Let's verify this hypothesis.
    # If .npy is minutes. `data[:, -1]/60` -> Hours.
    # Then `rt_mat2t` calculates `abs(item - term)`. If item is hours, `mat2t` is hours diff.
    # `t_dim` in `models.py` init is `hours+1` (24*7+1 = 169).
    # `MultiEmbed`: `traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1`. 
    # This clearly treats the input time as an index 1..168 (Hour of Week).
    
    # Wait. `process_traj` line 68: `data[:, -1] = np.array(data[:, -1]/60, dtype=int)`
    # If I provide minutes in .npy, then `load.py` converts it to Hours.
    # Then `process_traj` sorts by this time. 
    # Then `trajs` appends `user_traj`.
    # Then `MultiEmbed` does `% hours`. 
    # So yes, `load.py` expects **Minutes**.
    
    # So: Raw UTC -> Unix Timestamp (seconds) -> Minutes.
    # We can use minutes from epoch.
    # The format string '%a %b %d %H:%M:%S +0000 %Y' treats +0000 as a literal, resulting in naive datetime.
    # So we must use a naive timestamp for subtraction.
    base_time = pd.Timestamp("1970-01-01") 
    
    # To be safe and compatible with potential absolute time usage (though STAN seems relative or cyclic), 
    # let's just use "minutes from epoch".
    # 2012 is roughly 22 million minutes from 1970. Int32 fits.
    
    df['minutes'] = (df['dt'] - base_time) // pd.Timedelta('1min')
    
    # 4. Generate Output Dataframes
    # Data: [user_id, loc_id, minutes]
    output_data = df[['user_id', 'loc_id', 'minutes']].values.astype(np.int32)
    
    # POI: [loc_id, lat, lon]
    # We need unique POIs.
    poi_df = df[['loc_id', 'lat', 'lon']].drop_duplicates(subset=['loc_id']).sort_values('loc_id')
    output_poi = poi_df[['loc_id', 'lat', 'lon']].values.astype(np.float64)
    
    # 5. Save
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    data_path = os.path.join(args.output_dir, f"{args.dataset_name}.npy")
    poi_path = os.path.join(args.output_dir, f"{args.dataset_name}_POI.npy")
    
    np.save(data_path, output_data)
    np.save(poi_path, output_poi)
    
    print(f"Saved Check-ins to {data_path}, Shape: {output_data.shape}")
    print(f"Saved POIs to {poi_path}, Shape: {output_poi.shape}")

if __name__ == '__main__':
    args = parse_args()
    preprocess(args)
