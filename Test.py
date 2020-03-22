import glob
import numpy as np

from PIL import Image
from keras.models import load_model

caltech_dir = "dataset/image_test"
image_w = 100
image_h = 100

pixels = image_h * image_w * 3

X = []
filenames = []
files = glob.glob(caltech_dir + "/*.*")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    filenames.append(f)
    X.append(data)

model = load_model('./model/multi_img_classification.model')

prediction = model.predict(X)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
cnt = 0

for i in prediction:
    pre_ans = i.argmax()
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0:
        pre_ans_str = "음식"
    elif pre_ans == 1:
        pre_ans_str = "인물"
    elif pre_ans == 2:
        pre_ans_str = "도시_배경"
    else:
        pre_ans_str = "자연_배경"

    if i[0] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
    if i[1] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다.")
    if i[2] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
    if i[3] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다.")
    cnt += 1
