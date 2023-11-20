import os, time
os.system('cls')
appdata = os.getenv('APPDATA')

from gtts import gTTS
lang = 'en'

def speech(text:str):
    for f in os.listdir(appdata):
        if f[:3]=='tts' and f[-4:]=='.mp3':
            if int(f[3:-4])==1:
                i = int(f[3:-4])+1
            else:
                i = 1
            os.remove(appdata+'\\'+f)
    gTTS(text = text, lang=lang, slow=False).save(appdata+f'\\tts{i}.mp3')
    os.system(f'start {appdata}\\tts{i}.mp3')


# speech('Half of the calculations are done.')
# time.sleep(1.6)
speech('and 4 hours and 3 minutes are left')