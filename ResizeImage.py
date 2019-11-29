import os
from PIL import Image
with tf.device('/device:GPU:1'):
    
    for file in os.listdir('Audio_to_predict/'):
        img = Image.open('Audio_to_predict/' + file)
        new_img = img.crop((85, 50, 620, 382)) 
        new_img = new_img.resize((512,256))
        new_img.save('TestDataAudio/' + file, "PNG", optimize=True)