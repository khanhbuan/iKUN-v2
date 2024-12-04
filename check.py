import pandas as pd

motion_word = ["moving", "motion", "swift", "rapid", "quicker", "changing", "slowed", "stopping", "making", "came", "exceed", "used", "decelerating", "slowing", 
               "rapid", "faster", "speedier", "transferring", "outpace", "applied", "move", "relocating", "turning", "driving", "coming", "standing", "walking", 
               "movement", "transit", "parked", "parking", "parked,", "stationary", "going", "traveling", "turned"]

color_word = ["white", "blue", "yellow", "green", "silver", "red", "color", "black", "light"]

pos_word = ["direction", "front", "positioned", "located", "situated", "ahead", "right", "left", "before", "camera", "upright"]

if __name__ == "__main__":
    path = "./plugins/my_exp11_2/results/pedestrian_detailed.csv"
    data = pd.read_csv(path, delimiter=',', header=0)
    data = data.sort_values(by="HOTA___AUC")
    
    cases = {}

    for idx, item in data.iterrows():
        dir = item["seq"].split("+")
        if len(dir) == 1:
            continue
        video, seq = dir[0], dir[1]
        if video not in cases.keys():
            cases[video] = []
        score = float(item["HOTA___AUC"])
        cases[video].append((score, seq))

    data = {}
    keywords = {}

    for video in cases.keys():        
        for _, seq in cases[video]:
            if seq not in data.keys():
                data[seq] = []
                keywords[seq] = []
            exst = False
            for word in motion_word:
                if word in seq.split("-"):
                    data[seq].append(1)
                    keywords[seq].append(word)
                    exst = True
                    break
            for word in color_word:
                if word in seq.split("-"):
                    data[seq].append(2)
                    keywords[seq].append(word)
                    exst = True
                    break

            for word in pos_word:
                if word in seq.split("-"):
                    data[seq].append(3)
                    keywords[seq].append(word)
                    exst = True
                    break

            if len(data[seq]) == 0:
                data[seq].append(4)
    
    
    for seq in data.keys():
        if len(data[seq]) == 1 and data[seq][0] == 1:
            print(seq, keywords[seq])
    