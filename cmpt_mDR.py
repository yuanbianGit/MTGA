import os


def extract_map_sum(folder_path):
    # 存储所有文件的mAP值
    map_sum_att = 0.0
    map_sum = 0.0
    i = 0
    # 遍历文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            i = i+ 1
            # 打开文件并读取内容
            with open(file_path, 'r') as file:
                lines = file.readlines()
                if lines:  # 确保文件不为空
                    last_line = lines[-1].strip()  # 提取最后一行并去掉首尾空白
                    scecond_last_line = lines[-2].strip()  # 提取最后一行并去掉首尾空白
                    # 提取mAP数值
                    try:
                        map_value = float(scecond_last_line.split('mAP:')[1].split('%')[0].strip())
                        print(map_value)

                        map_sum += map_value

                        map_value_att = float(last_line.split('mAP:')[1].split('%')[0].strip())
                        print(map_value_att)
                        print("----")
                        map_sum_att += map_value_att
                    except (IndexError, ValueError):
                        print(f"文件 {file_name} 中无法提取 mAP 值")
    mean_map_att = map_sum_att/i
    mean_map = map_sum/i
    return mean_map_att, map_sum_att, map_sum, mean_map


# 文件夹路径
folder_path = '/data4/by/reid/github/MissRank_old/experiments/ours/meta_log_reivew_scal/c_M&D&T_viper'  # 替换为你的txt文件所在的文件夹路径
mean_map_att, map_sum_att, map_sum, mean_map = extract_map_sum(folder_path)
print(f"所有文件的mean_map_att为: {mean_map_att}")
print(f"所有文件的map_sum_att为: {map_sum_att}")
print(f"所有文件的map_sum为: {map_sum}")
print(f"所有文件的mean_map为: {mean_map}")
print("-----------------------------------")
print(f"所有文件的mDR为: {(mean_map-mean_map_att)/mean_map}")

