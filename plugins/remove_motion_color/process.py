import os
import shutil

if __name__ == "__main__":
    folders = ["0005", "0011", "0013"]
    root = "./results"

    motion_word = ["moving", "motion", "swift", "rapid", "quicker", "changing", "slowed", "stopping", "making", "came", "exceed", "used", "decelerating", "slowing", 
                   "rapid", "faster", "speedier", "transferring", "outpace", "applied", "move", "relocating", "turning", "driving", "coming", "standing", "walking", 
                   "movement", "transit", "parked", "parking", "parked,", "stationary", "going", "traveling", "turned", "walker", "walkers"]
    color_word = ["white", "blue", "yellow", "green", "silver", "red", "color", "black", "light", "direction", "opposite", "shirts" ,"shirt"]
    other_word = ["females", "women", "men", "males", "female", "sex", "womenfolk", 
                  "feminine", "gentlemen", "guys", "ladies"]
    pos_word = ["front", "positioned", "located", "situated", "ahead", "right", "left", "before", "camera", "upright"]

    for folder in folders:
        path = os.path.join(root, folder)
        for subfolder in os.listdir(path):
            subfolder2 = subfolder.split("-")
            exist = False
            for word in motion_word:
                if word in subfolder2:
                    exist=True
                    break
            for word in color_word:
                if word in subfolder2:
                    exist=True
                    break
            for word in other_word:
                if word in subfolder2:
                    exist=True
                    break
            
            ground_truth = os.path.join(path, subfolder, "gt.txt")
            prediction = os.path.join(path, subfolder, "predict.txt")
            os.remove(prediction)

            if exist:
                with open(prediction, 'w') as file:
                    pass
            else:
                print(subfolder2)
                shutil.copy(ground_truth, prediction)

                

