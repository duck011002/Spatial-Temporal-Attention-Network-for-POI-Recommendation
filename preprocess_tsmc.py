"""
将 TSMC2014 数据集转换为 STAN 项目所需的格式

输入: dataset_TSMC2014_NYC.txt (TSV格式, 8列)
输出: NYC.npy 和 NYC_POI.npy
"""

import numpy as np
from datetime import datetime
import os

def parse_time(time_str):
    """解析时间字符串"""
    try:
        return datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
    except:
        return None

def preprocess_tsmc2014(input_file, output_dir, dataset_name='NYC'):
    print(f"正在处理: {input_file}")
    
    users, venues, lats, lons, times = [], [], [], [], []
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i % 50000 == 0:
                print(f"已读取 {i} 行...")
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            dt = parse_time(parts[7])
            if dt is None:
                continue
            users.append(parts[0])
            venues.append(parts[1])
            lats.append(float(parts[4]))
            lons.append(float(parts[5]))
            times.append(dt)
    
    print(f"总记录数: {len(users)}")
    
    # ID映射 (从1开始)
    unique_users = sorted(set(users))
    unique_venues = sorted(set(venues))
    user_to_id = {u: i+1 for i, u in enumerate(unique_users)}
    venue_to_id = {v: i+1 for i, v in enumerate(unique_venues)}
    
    print(f"用户数: {len(unique_users)}, 地点数: {len(unique_venues)}")
    
    # 构建数据
    min_time = min(times)
    data = [[user_to_id[users[i]], venue_to_id[venues[i]], 
             int((times[i] - min_time).total_seconds() / 60)] 
            for i in range(len(users))]
    data = sorted(data, key=lambda x: (x[0], x[2]))
    data = np.array(data, dtype=np.int64)
    
    # 构建POI (平均坐标)
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
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'{dataset_name}.npy'), data)
    np.save(os.path.join(output_dir, f'{dataset_name}_POI.npy'), poi_data)
    
    print(f"\n=== 完成 ===")
    print(f"数据形状: {data.shape}, POI形状: {poi_data.shape}")
    print(f"用户ID: 1-{data[:,0].max()}, 地点ID: 1-{data[:,1].max()}")

if __name__ == '__main__':
    preprocess_tsmc2014(
        r'E:\dataset\dataset_tsmc2014\dataset_TSMC2014_NYC.txt',
        './data', 
        'NYC'
    )
