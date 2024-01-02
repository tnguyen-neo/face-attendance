from PIL import Image
import os, cv2

#Tao folder truoc khi thay doi destinationPath
destinationPath = r"D:\Fpt\Ky9(1421)\CapstoneProject\Code\Train-800_Test-200\Test\4. Tinh"
resourcePath = r"D:\Fpt\Ky9(1421)\CapstoneProject\Code\Train-800_Test-200\Test\4. Tinh_old"

def resizeImage(destPath, resPath):
    images = [f for f in os.listdir(resPath) if os.path.splitext(f)[-1] == '.jpg']
    count = 0
    for image in images:
        count += 1
        img = Image.open(os.path.join(resPath, image))
        img_resize = img.resize((224, 224), Image.ANTIALIAS)
        name = os.path.join(destPath+"\\" + str(count) + ".jpg")
        img_resize.save(name, 'JPEG', quality=95)
        print('done' + str(count))

resizeImage(destinationPath, resourcePath)