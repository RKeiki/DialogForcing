'''
cd /home/liujingqi/wanAR/OmniForcing/LTX-2 
.venv/bin/python packages/ltx-distillation/scripts/test_stage1_student_inference.py \
  --checkpoint /home/liujingqi/wanAR/OmniForcing/LTX-2/packages/ltx-distillation/outputs/stage1_bidirectional_dmd_split_teacher_worker/0402_103322_stage1_bidirectional_dmd_split_teacher_worker/checkpoint_000250/model.pt

常用可选参数：                                                      --num-prompts 50               
--start-index 0             
--device cuda:0           
--config-path /path/to/config.yaml   
--prompt-path /path/to/prompts.txt        
--output-dir /path/to/save_dir             

默认输出目录是该 checkpoint 目录下的 inference_first_050。
'''