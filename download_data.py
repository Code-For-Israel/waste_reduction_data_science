import gdown
import tarfile
import os

url = 'https://drive.google.com/uc?id=1GsC6vtBm47kNUPrCU-8_hXU1Pi0BraE9'
output = os.path.join(os.getcwd(), 'train.tar')
gdown.download(url, output, quiet=False)
tf = tarfile.open(output)
tf.extractall(os.getcwd())

url = 'https://drive.google.com/uc?id=1JVCAqZOhKCs3_5KrlZakB-ZJZTcyZSS3'
output = os.path.join(os.getcwd(), 'test.tar')
gdown.download(url, output, quiet=False)
tf = tarfile.open(output)
tf.extractall(os.getcwd())
