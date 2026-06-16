cd "C:\Users\tsah-05\OneDrive\Desktop\health_monitor"

$env:PATH = ".\.venv311\Lib\site-packages\nvidia\cublas\bin;.\.venv311\Lib\site-packages\nvidia\cudnn\bin;$env:PATH"

.\.venv311\Scripts\python.exe realtimetest.py --device cuda --compute-type float16 for running real time speech to text
(float32 bolgono)