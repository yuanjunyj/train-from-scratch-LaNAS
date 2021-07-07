import time
import pickle
import sys

def query_samples():
    samples_path = './OUTPUT/search_trajectory'
    try:
        with open(samples_path, "rb") as infile:
            print("Successfully read samples")
            samples = pickle.load(infile)
    except:
        print("File Not Exist")
        samples = {}
    
    return samples
    
def loop(func, interval):
    while True:
        curr_res = func()
        print("Current Samples Count:", len(curr_res))
        print(curr_res)
        sys.stdout.flush()
        if len(curr_res) > 1000:
            break
        time.sleep(interval)
    
if __name__ == "__main__":
    loop(query_samples, 60 * 60)
