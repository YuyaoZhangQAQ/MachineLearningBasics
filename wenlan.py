from numpy import source
import requests
url = 'http://buling.wudaoai.cn/retrieval'

files = {'local_img': open('Order5.png','rb')}

params = {'source' : 'album'}

r = requests.post(url, files=files, params=params)

r.json()