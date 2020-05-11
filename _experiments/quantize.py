from PIL import Image
import numpy as np

img = Image.open("../counter_images/01305.png").convert('gray')
imgQ = img.quantize(3)

img.show()
