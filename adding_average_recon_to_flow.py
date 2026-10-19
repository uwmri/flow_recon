import h5py
from pathlib import Path
import argparse

def adding_average_to_flow(flow_file, average_file, path):
    
    flow_path = path / Path(flow_file)
    avg_path = path / Path(average_file)
    print(avg_path)

    avg_dataset = ["CD", "MAG", "comp_vd_1", "comp_vd_2", "comp_vd_3"]

    with h5py.File(avg_path, 'r') as src, \
        h5py.File(flow_path, 'r+') as dst:
            
            src_data = src['/Data']
            dst_data = dst['/Data']
            
            for name in avg_dataset:
                if name not in src_data:
                    raise KeyError(f"Missing {name} in {avg_path}")

            for name in avg_dataset:
                if name in dst_data:
                    del dst_data[name]
                    
                src_data.copy(name, dst_data, name=name)
                
                print(f"Replaced {name}: {dst_data[name].shape}")
                
    print(f"Finished updating {flow_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--flow_file', required=True)
    parser.add_argument('-a', '--avg_file', required=True)
    parser.add_argument('-p', '--path', required=True)
    args = parser.parse_args()
    
    adding_average_to_flow(flow_file = args.flow_file, average_file=args.avg_file, path=args.path)