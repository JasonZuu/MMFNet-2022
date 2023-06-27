from video_tools.Face_Recognition import FR_system
from video_tools.Extract_features import EF_system, batch_process
import os
import shutil

def get_dataset(datas_dir, save_dir):
    Face_recognizer = FR_system(save_dir=save_dir)
    for root, _, files in os.walk(datas_dir, topdown=True):
        for file_path in files:
            file_path = os.path.join(root, file_path)
            Face_recognizer.face_recognize(file_path, 10, 0.44)
            shutil.copy(file_path, Face_recognizer.save_video_dir)
            print ("copy %s -> %s"%(file_path, Face_recognizer.save_video_dir))
            Face_recognizer.clear()
    Features_extractor = EF_system()
    batch_process(Features_extractor, save_dir, f"{save_dir}/Features.csv")

if __name__ == "__main__":
    get_dataset("./active_100", "./dataset/active_100")
    