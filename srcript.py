import re
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    counts = {}
    num_runs = 0
    try:
        with open(args.filename) as f:
            line = f.readline()
            while line:
                match = re.findall('"conclusion": "(\w+)"', line)

                for result in match:
                    num_runs += 1
                    if result in counts:
                        counts[result]+=1
                    else:
                        counts[result]=1

                line = f.readline()

        for key in counts:
            print(f"{key:16}: {counts[key]:06d} => {round(counts[key]/num_runs*100,2):3.2f}%")
    except:
        print("Can not open file", args.filename)


