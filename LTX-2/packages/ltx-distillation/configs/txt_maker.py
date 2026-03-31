import csv
import random

input_csv = "/home/liujingqi/wanAR/mtv_meta/mtv_merged.csv"      # 原始CSV文件
output_txt = "output.txt"    # 输出TXT文件
target_column = "caption_full"       # 要保留的字段名

values = []

with open(input_csv, "r", encoding="utf-8-sig", newline="") as f_in:
    reader = csv.DictReader(f_in)

    if target_column not in reader.fieldnames:
        raise ValueError(f"字段 {target_column} 不存在于CSV中，可用字段: {reader.fieldnames}")

    for row in reader:
        values.append(str(row[target_column]).strip())

random.shuffle(values)

with open(output_txt, "w", encoding="utf-8") as f_out:
    for value in values:
        f_out.write(value + "\n")

print(f"已保存到 {output_txt}")