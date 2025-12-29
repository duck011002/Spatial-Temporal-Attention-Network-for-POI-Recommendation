"""
统一数据预处理脚本
将 TSMC2014 和 poidata 数据集转换为 STAN 所需格式

输出格式:
- {name}.npy: (N, 3) -> [user_id, loc_id, time_in_minutes]
- {name}_POI.npy: (L, 3) -> [loc_id, latitude, longitude]
"""

import numpy as np
from datetime import datetime
import os
import re

# ============== TSMC2014 格式 ==============
# 8列TSV: user_id, venue_id, category_id, category_name, lat, lon, tz_offset, utc_time
# 时间格式: "Tue Apr 03 18:00:09 +0000 2012"

def parse_tsmc_time(time_str):
    try:
        return datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
    except:
        return None

def process_tsmc2014(input_file, output_dir, dataset_name):
    """处理 TSMC2014 数据集 (NYC, TKY)"""
    print(f"\n{'='*50}")
    print(f"处理 TSMC2014: {dataset_name}")
    print(f"{'='*50}")
    
    users, venues, lats, lons, times = [], [], [], [], []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print(f"已读取 {i} 行...")
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            dt = parse_tsmc_time(parts[7])
            if dt is None:
                continue
            users.append(parts[0])
            venues.append(parts[1])
            lats.append(float(parts[4]))
            lons.append(float(parts[5]))
            times.append(dt)
    
    return build_and_save(users, venues, lats, lons, times, output_dir, dataset_name)


# ============== poidata 格式 ==============
# 5列TSV: user_id, loc_id, "lat,lon", time(HH:MM), date_id
# 例: USER_42  LOC_15693  37.616,-122.386  21:59  0

def process_poidata(input_dir, output_dir, dataset_name):
    """处理 poidata 数据集 (Gowalla, Foursquare)"""
    print(f"\n{'='*50}")
    print(f"处理 poidata: {dataset_name}")
    print(f"{'='*50}")
    
    users, venues, lats, lons, times = [], [], [], [], []
    
    # 合并 train/tune/test
    for split in ['train.txt', 'tune.txt', 'test.txt']:
        filepath = os.path.join(input_dir, split)
        if not os.path.exists(filepath):
            continue
        print(f"读取 {split}...")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                user_id = parts[0].replace('USER_', '')
                loc_id = parts[1].replace('LOC_', '')
                
                # 解析坐标 "lat,lon"
                coord_parts = parts[2].split(',')
                if len(coord_parts) != 2:
                    continue
                try:
                    lat = float(coord_parts[0])
                    lon = float(coord_parts[1])
                except:
                    continue
                
                # 解析时间 "HH:MM" 和 date_id
                time_parts = parts[3].split(':')
                date_id = int(parts[4])
                if len(time_parts) != 2:
                    continue
                
                # 转换为分钟: date_id * 24 * 60 + hour * 60 + minute
                hour, minute = int(time_parts[0]), int(time_parts[1])
                time_minutes = date_id * 24 * 60 + hour * 60 + minute
                
                users.append(user_id)
                venues.append(loc_id)
                lats.append(lat)
                lons.append(lon)
                times.append(time_minutes)
    
    return build_and_save_poidata(users, venues, lats, lons, times, output_dir, dataset_name)


def build_and_save(users, venues, lats, lons, times, output_dir, dataset_name):
    """构建并保存数据 (TSMC2014格式，times是datetime)"""
    print(f"总记录数: {len(users)}")
    
    if len(users) == 0:
        print("警告: 没有有效数据!")
        return
    
    # ID映射 (从1开始)
    unique_users = sorted(set(users))
    unique_venues = sorted(set(venues))
    user_to_id = {u: i+1 for i, u in enumerate(unique_users)}
    venue_to_id = {v: i+1 for i, v in enumerate(unique_venues)}
    
    print(f"用户数: {len(unique_users)}, 地点数: {len(unique_venues)}")
    
    # 时间转分钟
    min_time = min(times)
    
    # 构建数据
    data = [[user_to_id[users[i]], venue_to_id[venues[i]], 
             int((times[i] - min_time).total_seconds() / 60)] 
            for i in range(len(users))]
    data = sorted(data, key=lambda x: (x[0], x[2]))
    data = np.array(data, dtype=np.int64)
    
    # 构建POI
    venue_coords = {}
    for i, vid in enumerate(venues):
        if vid not in venue_coords:
            venue_coords[vid] = {'lats': [], 'lons': []}
        venue_coords[vid]['lats'].append(lats[i])
        venue_coords[vid]['lons'].append(lons[i])
    
    poi_data = [[venue_to_id[vid], 
                 np.mean(venue_coords[vid]['lats']), 
                 np.mean(venue_coords[vid]['lons'])] 
                for vid in unique_venues]
    poi_data = sorted(poi_data, key=lambda x: x[0])
    poi_data = np.array(poi_data, dtype=np.float64)
    
    # 保存
    save_data(data, poi_data, output_dir, dataset_name)


def build_and_save_poidata(users, venues, lats, lons, times, output_dir, dataset_name):
    """构建并保存数据 (poidata格式，times已是分钟)"""
    print(f"总记录数: {len(users)}")
    
    if len(users) == 0:
        print("警告: 没有有效数据!")
        return
    
    # ID映射 (从1开始)
    unique_users = sorted(set(users), key=lambda x: int(x) if x.isdigit() else x)
    unique_venues = sorted(set(venues), key=lambda x: int(x) if x.isdigit() else x)
    user_to_id = {u: i+1 for i, u in enumerate(unique_users)}
    venue_to_id = {v: i+1 for i, v in enumerate(unique_venues)}
    
    print(f"用户数: {len(unique_users)}, 地点数: {len(unique_venues)}")
    
    # 构建数据
    data = [[user_to_id[users[i]], venue_to_id[venues[i]], times[i]] 
            for i in range(len(users))]
    data = sorted(data, key=lambda x: (x[0], x[2]))
    data = np.array(data, dtype=np.int64)
    
    # 构建POI
    venue_coords = {}
    for i, vid in enumerate(venues):
        if vid not in venue_coords:
            venue_coords[vid] = {'lats': [], 'lons': []}
        venue_coords[vid]['lats'].append(lats[i])
        venue_coords[vid]['lons'].append(lons[i])
    
    poi_data = [[venue_to_id[vid], 
                 np.mean(venue_coords[vid]['lats']), 
                 np.mean(venue_coords[vid]['lons'])] 
                for vid in unique_venues]
    poi_data = sorted(poi_data, key=lambda x: x[0])
    poi_data = np.array(poi_data, dtype=np.float64)
    
    # 保存
    save_data(data, poi_data, output_dir, dataset_name)


def save_data(data, poi_data, output_dir, dataset_name):
    """保存数据到指定目录"""
    os.makedirs(output_dir, exist_ok=True)
    
    data_file = os.path.join(output_dir, f'{dataset_name}.npy')
    poi_file = os.path.join(output_dir, f'{dataset_name}_POI.npy')
    
    np.save(data_file, data)
    np.save(poi_file, poi_data)
    
    print(f"\n已保存:")
    print(f"  {data_file} - shape: {data.shape}")
    print(f"  {poi_file} - shape: {poi_data.shape}")
    print(f"  用户ID范围: 1-{data[:,0].max()}")
    print(f"  地点ID范围: 1-{data[:,1].max()}")


def main():
    # 基础路径
    base_output = './data'
    
    # 1. 处理 TSMC2014 数据集
    tsmc_dir = './dataset_tsmc2014'
    if os.path.exists(tsmc_dir):
        # NYC
        nyc_file = os.path.join(tsmc_dir, 'dataset_TSMC2014_NYC.txt')
        if os.path.exists(nyc_file):
            process_tsmc2014(nyc_file, os.path.join(base_output, 'TSMC_NYC'), 'NYC')
        
        # TKY
        tky_file = os.path.join(tsmc_dir, 'dataset_TSMC2014_TKY.txt')
        if os.path.exists(tky_file):
            process_tsmc2014(tky_file, os.path.join(base_output, 'TSMC_TKY'), 'TKY')
    
    # 2. 处理 poidata 数据集
    poidata_dir = './poidata'
    if os.path.exists(poidata_dir):
        # Gowalla
        gowalla_dir = os.path.join(poidata_dir, 'Gowalla')
        if os.path.exists(gowalla_dir):
            process_poidata(gowalla_dir, os.path.join(base_output, 'Gowalla'), 'Gowalla')
        
        # Foursquare
        fsq_dir = os.path.join(poidata_dir, 'Foursquare')
        if os.path.exists(fsq_dir):
            process_poidata(fsq_dir, os.path.join(base_output, 'Foursquare'), 'Foursquare')
    
    print(f"\n{'='*50}")
    print("全部处理完成!")
    print(f"{'='*50}")
    print("\n生成的数据目录结构:")
    print("data/")
    print("├── TSMC_NYC/      # NYC.npy, NYC_POI.npy")
    print("├── TSMC_TKY/      # TKY.npy, TKY_POI.npy")
    print("├── Gowalla/       # Gowalla.npy, Gowalla_POI.npy")
    print("└── Foursquare/    # Foursquare.npy, Foursquare_POI.npy")


if __name__ == '__main__':
    main()
