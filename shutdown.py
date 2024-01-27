
import time, os, sys
from survey_stats.keyboard import Keyboard
from survey_stats.sound import Sound



def shutdown(minutes:int = 10, volume_up:bool=True, scaped:bool=False):
    import winsound
    t = minutes*60
    print('shutting down...')
    if volume_up:
        Sound.volume_up()
    while t>=0:
        print(str(t).ljust(2), end='\r')
        t -= 2
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        with open(sys.argv[0]) as f:
            file_text = f.read()
        start_loc = file_text.find('shutdown(')
        end_loc = file_text[start_loc:].find(')')
        args = file_text[start_loc+9:start_loc+end_loc].split(',')
        for arg in args:
            if 'scaped=' in arg:
                scaped = arg[8:]=='True'
                break
        else:
            if len(args)==3:
                scaped = args[2].strip()=='True'
        if scaped:
            break
    else:
        import subprocess
        subprocess.run(["shutdown", "-s"])


if __name__=='__main__':
    # shutdown(5*60)
    pass