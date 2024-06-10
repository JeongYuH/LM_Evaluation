from Eval_json import Evaluation_json
import datetime

save_time = datetime.datetime.now().strftime('%y%m%d_%H:%M:%S')



file1 = 'Evaluation\\Example_input\\test_output_gpt.json'
file2 = 'Evaluation\\Example_input\\test_output_PAL_orbit_v0.2.2.2.json'
result = 'Evaluation\\Result\\' + datetime.datetime.now().strftime('%y%m%d_%H%M%S') + '.json' # Default로 설정된 파일명 -> 원하는대로 수정하여 사용하기.

eval = Evaluation_json(file_path1=file1, file_path2=file2, question_index=3)

# eval.run(filepath=result)   # 경로를 설정하여 사용할 경우
eval.run()                      # 경로 설정 안해도됨.