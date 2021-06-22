import gdown
import tarfile

url = 'https://drive.google.com/uc?id=1GsC6vtBm47kNUPrCU-8_hXU1Pi0BraE9'
output = r'/home/yotam/facemask_obj_detect/train.tar'
gdown.download(url, output, quiet=False)
tf = tarfile.open(output)
tf.extractall(r'/home/yotam/facemask_obj_detect/')

url = 'https://drive.google.com/uc?id=1JVCAqZOhKCs3_5KrlZakB-ZJZTcyZSS3'
output = r'/home/yotam/facemask_obj_detect/test.tar'
gdown.download(url, output, quiet=False)
tf = tarfile.open(output)
tf.extractall(r'/home/yotam/facemask_obj_detect/')
