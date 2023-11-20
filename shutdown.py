
import time, os

def shutdown(minutes:int = 10):
    import winsound
    t = minutes*60
    print('shutting down...')
    while t>=0:
        print(str(t).ljust(2), end='\r')
        t -= 2
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
        # time.sleep(1)

    import subprocess
    subprocess.run(["shutdown", "-s"])

if __name__=='__main__':
    shutdown(5*60)