from PIL import Image
import os

list = os.listdir("./linghsi")
print(list)
i = 1
for image in list:
    id_tag = image.find(".")
    name = image[0:id_tag]
    print(name)

    im = Image.open("./linghsi/" + image)
    out = im.resize((128, 128))
    # out.show()
    out.save("./tem/" + '-' + str(i) + ".jpg")
    i = i + 1
