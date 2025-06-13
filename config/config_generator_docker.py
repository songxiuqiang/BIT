import os
import yaml
import numpy as np
# import trimesh
# import pyvista as pv
# from scipy.spatial import Delaunay

from glob import glob

# 获取当前目录下所有的文件夹名字
def GetFileName(filename):
    for root, dirs, files in os.walk(filename):
        # print(root)
        # print(files)
        # print(dirs)
        return dirs
        
def GetSubDirs(path,mode='*'):
	files = glob(path + mode, recursive=True)
	files = sorted(files)
	return files

def GenRbotYmls(scene, models, rbotRoot, saveRoot):
    """
    Generate yaml files of rbot dataset
    Args:
    	models (list): a list of models' name
        rbotRoot (string): path of rbot dataset 
        saveRoot (string): path of save generated yaml files
    """
    for model in models:
        content = {
            "root": rbotRoot,
            "scene": scene,
            "modelName": model,
            # "modelPath": rbotRoot + model + "/" + model + "_simple.obj",
            # "modelPath": rbotRoot + "RBOT-model-coarse-02/" + model + "-coarse.obj",
            "modelPath": "/home/songxiuqiang/Documents/academic conference/2025-CVPR/RBOT-results/3d12r12_angle3/" + model + "/model1.obj",
            # "modelPath": "/home/songxiuqiang/dataset/RBOT/models_CGF/3/" + model + ".obj",
            # "modelPath": "/home/songxiuqiang/Documents/academic conference/2025-AAAI/datas/RBOT-results/3d3r6_angle3/" + model + "/model.obj",
            "frames": rbotRoot + model + "/frames/",
            "depth": rbotRoot + model + "/depth/",
            "gt": rbotRoot + "poses_first.txt"
        }
        savePath = saveRoot + scene + "_" + model + ".yml"
        with open(savePath, "w") as f:
            yaml.safe_dump(data=content, stream=f)
        with open(savePath, "r+") as f:
            lines = f.read()
            linesQuotes = lines.replace(': ', ': \"')
            linesQuotes = linesQuotes.replace('\n', '\"\n')
            f.seek(0, 0)
            f.write("%YAML:1.0\n \n" + linesQuotes)

def GenRbotYmls4Summer(scene, models, rbotRoot, saveRoot):
    print("Generating Rbot ymls for summer")
    for model in models:
        yml_path = saveRoot + scene + "_" + model + ".yml" 
        print("yml path", yml_path)
        with open(yml_path, 'w') as yml_file:
            yml_file.write(f"%YAML:1.0\n\n")
            yml_file.write(f"dataset: \"rbot\"\n")
            yml_file.write(f"root: \"{rbotRoot}{model}/\"\n")
            yml_file.write(f"cmc: \"cmc/\"\n")
            yml_file.write(f"frames: \"frames/\"\n")
            yml_file.write(f"width: 640\n")
            yml_file.write(f"height: 512\n")
            yml_file.write(f"angle: 3.0\n")
            yml_file.write(f"mask: \"mask/\"\n")
            yml_file.write(f"gt: \"{rbotRoot}poses_first.txt\"\n")
            yml_file.write(f"modelName: \"{model}\"\n")
            yml_file.write(f"modelPath: \"{rbotRoot}{model}/{model}_simple.obj\"\n")
            yml_file.write(f"scaledModelPath: \"deformed_model_rescaled/model.obj\"\n")
            yml_file.write(f"scene: \"{scene}\"\n\n")
            yml_file.write(f"K: \"config/camera/rbot_camera.yml\"\n\n")
            yml_file.write(f"reference_imgs_nums: 7\n")
            yml_file.write(f"estimated_imgs_nums: 6\n")
            yml_file.write(f"max_model_deformed_nums: 3\n\n")
            yml_file.write(f"# 可微分渲染器配置\n")
            yml_file.write(f"renderer_img_size: 256\n")
            yml_file.write(f"iteration_nums_in_each_deform: 200\n")
            yml_file.write(f"batch_size: 6\n")
            yml_file.write(f"viewing_angle: 30\n\n")
            yml_file.write(f"template_mesh: \"sphere_2600_unit.obj\"\n")
            yml_file.write(f"output_dir: \"{rbotRoot}{model}/cmc/deformed_model\"\n\n")
            yml_file.write(f"save_obj_path: \"\"\n")
            yml_file.write(f"save_index: 0\n")
            yml_file.write(f"tracker: \"tracker/bin/eg_summer\"\n\n")
            yml_file.write(f"communication: \n")
            yml_file.write(f"    root: \"{rbotRoot}{model}/cmc/\"\n\n")
            yml_file.write(f"sam: \n")
            yml_file.write("    checkpoint: \"sam_vit_b_01ec64.pth\"\n")
            yml_file.write("    model: \"vit_b\"\n")
            yml_file.write("    device: \"cuda\"\n\n")
            yml_file.write(f"show: \n")
            yml_file.write("    mask: False\n")
            yml_file.write("    bbx: False\n")
            yml_file.write("    points: False\n\n")
            yml_file.write(f"save: \n")
            yml_file.write("    dir: \"\"\n")
            yml_file.write("    mask: False\n")
            yml_file.write("    bbx: False\n")
            yml_file.write("    points: False\n")    
           
            
def GenMopedConfigs(moped_root, generated_ymls_root):
    models = ["black_drill", "duplo_dude", "graphics_card", "orange_drill", "remote", \
             "toy_plane", "cheezit", "duster", "moped.tgz", "pouch", "rinse_aid", "vim_mug"]
    sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", \
                 "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    types = ["reference", "evaluation"]
    
    for i in range(0,2):  
        for m in models:
            print("model", m)
            refRoot = moped_root + m + "/reference/" 
            root = moped_root + m + "/" + types[i] + "/" 
            mesh =  moped_root + m + "/reference/integrated_raw.obj"
            
            # 检查参考目录是否存在
            if not os.path.exists(root):
                print(f"Directory does not exist: {root}")
                continue  # 跳过不存在的目录 
            
            # 获取ref_dir下的文件夹名称dirs，不包括文件
            dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
            print(dirs)
            
            for dir in dirs:
                if dir not in sequences:
                    continue
                # 创建 generated_ymls_root/m/reference/dir.yml
                yml_dir = os.path.join(generated_ymls_root, m, types[i])
                os.makedirs(yml_dir, exist_ok=True)
                yml_file_path = os.path.join(yml_dir, f"{dir}.yml")
                print("yml_file_path", yml_file_path)
                
                with open(yml_file_path, 'w') as yml_file:
                    yml_file.write(f"%YAML:1.0\n\n")
                    yml_file.write(f"dataset: \"moped\"\n")
                    yml_file.write(f"root: \"{root}{dir}/\"\n")
                    yml_file.write(f"refRoot: \"{refRoot}00/\"\n")
                    yml_file.write(f"cmc: \"cmc/\"\n")
                    yml_file.write(f"frames: \"color/\"\n")
                    yml_file.write(f"mask: \"mask/\"\n")
                    yml_file.write(f"angle: 3.0\n")
                    yml_file.write(f"width: 640\n")
                    yml_file.write(f"height: 480\n")
                    yml_file.write(f"gt: \"scene/trajectory.log\"\n")
                    yml_file.write(f"modelName: \"{m}-{dir}\"\n")
                    yml_file.write(f"modelPath: \"{mesh}\"\n")
                    yml_file.write(f"scaledModelPath: \"cmc/deformed_model_rescaled/model.obj\"\n")
                    yml_file.write(f"scene: \"\"\n\n")
                    yml_file.write(f"K: \"{root}{dir}/intrinsics.json\"\n\n")
                    yml_file.write(f"reference_imgs_nums: 1\n")
                    yml_file.write(f"estimated_imgs_nums: 3\n")
                    yml_file.write(f"max_model_deformed_nums: 3\n\n")
                    yml_file.write(f"# 可微分渲染器配置\n")
                    yml_file.write(f"renderer_img_size: 256\n")
                    yml_file.write(f"iteration_nums_in_each_deform: 200\n")
                    yml_file.write(f"batch_size: 6\n")
                    yml_file.write(f"viewing_angle: 30\n\n")
                    yml_file.write(f"template_mesh: \"sphere_2600_unit.obj\"\n")
                    yml_file.write(f"output_dir: \"{root}{dir}/cmc/deformed_model\"\n\n")
                    yml_file.write(f"save_obj_path: \"\"\n")
                    yml_file.write(f"save_index: 0\n")
                    yml_file.write(f"tracker: \"tracker/bin/eg_summer\"\n\n")
                    yml_file.write(f"communication: \n")
                    yml_file.write(f"    root: \"{root}{dir}/cmc/\"\n\n")
                    yml_file.write(f"sam: \n")
                    yml_file.write("    checkpoint: \"sam_vit_b_01ec64.pth\"\n")
                    yml_file.write("    model: \"vit_b\"\n")
                    yml_file.write("    device: \"cuda\"\n\n")
                    yml_file.write(f"show: \n")
                    yml_file.write("    mask: False\n")
                    yml_file.write("    bbx: False\n")
                    yml_file.write("    points: False\n\n")
                    yml_file.write(f"save: \n")
                    yml_file.write("    dir: \"\"\n")
                    yml_file.write("    mask: False\n")
                    yml_file.write("    bbx: False\n")
                    yml_file.write("    points: False\n")    

            
    
if __name__ == '__main__':

    # RBOT Dataset for BIT 
    if 1:
        models = ["ape", "bakingsoda", "benchviseblue", "broccolisoup", "cam", "can", "cat", "clown", \
                  "cube", "driller", "duck", "eggbox", "glue", "iron", "koalacandy", "lamp", "phone", "squirrel"]
        rbotRoot = "/home/songxiuqiang/dataset/RBOT/"
        r4s_saveRoot = "rbot/"
        scene = "a_regular"
        GenRbotYmls4Summer(scene, models, rbotRoot, r4s_saveRoot)

    # RBOT Dataset for test other trackers
    if 1:
        models = ["ape", "bakingsoda", "benchviseblue", "broccolisoup", "cam", "can", "cat", "clown", \
                  "cube", "driller", "duck", "eggbox", "glue", "iron", "koalacandy", "lamp", "phone", "squirrel"]
        rbotRoot = "/home/songxiuqiang/dataset/RBOT/"
        saveRoot = "rbot_3d3r6/"
        GenRbotYmls("a_regular", models, rbotRoot, saveRoot)
        GenRbotYmls("b_dynamiclight", models, rbotRoot, saveRoot)
        GenRbotYmls("c_noisy", models, rbotRoot, saveRoot)
        GenRbotYmls("d_occlusion", models, rbotRoot, saveRoot)

    
	# moped dataset
    if 1:
        moped_root_path = "../dataset/moped/"
        moped_generated_ymls_path = "moped/"
        GenMopedConfigs(moped_root = moped_root_path, generated_ymls_root = moped_generated_ymls_path)
        