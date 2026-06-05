from mygaze_hook import MyGazeHook

hook = MyGazeHook()
hook.connect()

for i in range(20):
    pt = hook.get_gaze_point()
    pd = hook.get_pupil_diameter()
    print(f"Gaze: {pt}  |  Pupil: {pd}")
    import time; time.sleep(0.1)

hook.disconnect()