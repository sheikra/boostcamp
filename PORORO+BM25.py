#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


get_ipython().system('pip install datasets==1.5.0')
get_ipython().system('pip install konlpy')
get_ipython().system('pip install pororo')
get_ipython().system('pip install python-mecab-ko')
get_ipython().system('pip install rank_bm25')


# In[ ]:


get_ipython().system('git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git')


# In[ ]:


cd Mecab-ko-for-Google-Colab


# In[ ]:


get_ipython().system('bash install_mecab-ko_on_colab190912.sh')


# In[ ]:


get_ipython().system('mkdir /content/data')
get_ipython().system('unzip /content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/data.zip -d /content/data')
get_ipython().system('rm -r /content/data/__MACOSX')


# In[ ]:


import zipfile
from datasets import load_from_disk
import json
import pandas as pd
import re
from konlpy.tag import Mecab
# from konlpy.tag import Okt
# from khaiii import KhaiiiApi
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from pororo import Pororo
from collections import OrderedDict
from collections import Counter
from collections import defaultdict
import operator


# In[ ]:


import os
import json
from tqdm.notebook import tqdm
import string
import pickle


# In[ ]:


f = zipfile.ZipFile('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/data.zip')
f.extractall('/content')
f.close()


# In[ ]:


test_dataset = load_from_disk('/content/data/test_dataset/validation')
print(test_dataset)


# In[ ]:


with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/title_고려안함/wiki_for_retriever/all_wikipedia_documents.json', 'r') as f:
    wiki_data = pd.DataFrame(json.load(f)).transpose()


# In[ ]:


wiki_data['total'] = wiki_data['title'] + ' ' + wiki_data['text']

wiki_data['total'] = wiki_data['total'].apply(lambda x : x.replace('\\n\\n',' '))
wiki_data['total'] = wiki_data['total'].apply(lambda x : x.replace('\n\n',' '))
wiki_data['total'] = wiki_data['total'].apply(lambda x : x.replace('\\n',' '))
wiki_data['total'] = wiki_data['total'].apply(lambda x : x.replace('\n',' '))
wiki_data['total'] = wiki_data['total'].apply(lambda x : ' '.join(re.sub(r'[^0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', ' ', str(x.lower().strip())).split()))


# In[ ]:


mecab = Mecab()
tokenizer = Pororo(task="tokenization", lang="ko", model="mecab.bpe64k.ko")


# In[ ]:


wiki_data['total'] = wiki_data['total'].map(tokenizer)


# In[ ]:


tokenized_corpus = wiki_data['total'].to_list()


# In[ ]:


len(tokenized_corpus)


# In[ ]:


dummy_data = [''] * (67277 - 56737) #  67277 - 56737 는 KorQuAD 1.0로 추가된 문서의 수
len(dummy_data)


# In[ ]:


bm25_ori = BM25Plus(tokenized_corpus)


# # 비교를 위한 기본 모델

# In[ ]:


# answer = OrderedDict()
# for num in tqdm(range(len(test_dataset))):
#     id = test_dataset['id'][num]
#     query = test_dataset['question'][num]
#     tokenized_query = tokenizer(query)
    
#     for doc in bm25_ori.get_top_n(tokenized_query, wiki_data['text'].to_list()[:56737]+dummy_data, n=30):
#         if doc == '':
#             pass
#         ans = mrc(query,doc)[0]
#         if ans == '' or len(ans) == 1 or len(ans) >= 100: 
#             pass
#         else:
#             answer[id] = ans
#             break

# with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/predictions_0506비교용.json', 'w') as f:
#     json.dump(answer, f)


# ## Word Sense Disambiguation
# 

# In[ ]:


wsd = Pororo(task="wsd", lang="ko")


# # 오답문장이 가지는 바이브가 있다. '다.' 를 포함하는 문장 제거

# In[ ]:


# answer = OrderedDict()
# for num in tqdm(range(len(test_dataset))):
#     id = test_dataset['id'][num]
#     query = test_dataset['question'][num]
#     tokenized_query = tokenizer(query)
    
#     for doc in bm25_ori.get_top_n(tokenized_query, wiki_data['text'].to_list()[:56737]+dummy_data, n=30):
#         if doc == '':
#             pass

#         ans = mrc(query,doc)[0]
#         if (len(ans) <= 1): # 기존에는 if else에서 else쪽으로 로 처리했는데, 여기는 그럴 수 없음
#             continue

#         if '다.' in ans:
#             continue

#         if ans[-1] == '의': # '가', '이' 
#             wsd_result = wsd(ans)[-1]
#             if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의'):
#                 ans = ans[:-1]
#                 if (len(ans) <= 1): 
#                     continue
#         print(ans)
#         answer[id] = ans
#         break

# with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/predictions_0506_의와다쩜재낌.json', 'w') as f:
#     json.dump(answer, f)


# ## 결론: 긴 오답 문장을 자연스럽게 제거할 수 있다. '며,' '고,' 도 필요할 것이다.

# ![image.png](data:image/png;base64,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)

# ![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAAhCAYAAACvFxNAAAAgAElEQVR4Ae2dd3sTV9Pw860kmRYIJRBK6Am9J4QaSiihhtz0EoqBgOm9B+yEjrEBE/cm+evMc/1mzuyuhIyd3Hmf8t77x167Wkk7Z+ZMP3NmP8lmc5LNZqMjl81KTXWNdHR2SldXl+TzeSkUCtKTL0h3oSDtHe1y8uRJyed7ZPjw4fq/TDYrU6dO1utDhw7J+4b30sl/u/PS2dkhb96+lYcPH8r69esNTi4jwDG4Bn/Pnj3S3d2t93PZnPDM5LgmjBsvU6dPlxOVJ/R+JhOPm9/+8sth6ezstP9kspLN5SSbyUpHR7scP2H/iZ6Xy8jno0YpXgsWLCiCw7jKwc9lspLhmTkbVyn8XC4TPycBP5vNfIBLtgz+PrYUfkr/lP8+lP9U/lL9k+rff97+fOKGJ5ONDdj0r6bL8mXL5PeaGtn5004ZN26cDBgwIDJw33zzjeQLeRkxwhwAf8aFCxfUqN6+fUt+/vlfsnTpt7J79265d++e9BQKcuyXo2ZAMZBq4GOYu/fsVgdA72NkM8FJCAaX+1u3blWHhGvGm0t8h3NRU1OTeC4OQkY6OjqktbVVXtfWyutXtVL7ulZqX9VKfX29FAp5cQcgib8a+TLwbczBASiBn/wOuOZYxfipYVfHANw+xD+FH9MqpX95/k/yWCn/J79L+S+Vv1T/mP53uUj1b1Zwok23xvbnEzdGRB1GrDhi7erukoMHD1q0nsPo2m+GfjpU1q1bJwMHxk4BUfj79+/lyZMnEZDYQOek9vVrqa2tVRhLliyRZ8+ey9Nnz+TZ02fy7NkzaWpsVOeBa73Pd8+fyaNHjyL4W7ZuUweAyWSs8fMZG5FjQEwdCPseB+DVq1qpPHlSMxdkLzjOnTsv+UJBFsy3DEA5/CFWJuEIOP4Opz/wjaYe0SSMnDoxsTOQwv+Q/1L6p/yXyl8cCKX6x3REqn/7b//6sj9RBkA9g5AFcGPU1dWp0fuWrVtky5YfZcuWLbJ1y1bZuoXPW2TSxEnBaTBDdunCJcnnu+X69euyfccOWbRooez6aZfcvXtHCj0FOXLkFzXSM77+Wi5evBgdF7i+xOcLeu/SxUvhuwtqrH1sZAB6Cj36/OvXr8n1a9ftuG7nG9duyMhRI3VMtoSQkY72Drl3766sXLVSVq5cJatWrpSVK1bI5h83Fy8BaDbBDLTjb5FU8JoStHEGdOdJiZzx1LU9w+HzG3UU9Pk5xR+lZji5J/YhDJu44DCUjC2FH+gWZVKM/5g3z6Sk9IdGGc02pfzn8pXKH7oj1T+p/nX78wkXkZftaXnWs3NZqaurl+amZmluadJzU3OzcOjn5mbZ9ONmyQXDiMHiet++ffL27VtNvRfyeeno6JT6unrZsX27OQs5j9Rtjf5j8EljWdrCDOGcuXPlUXW1VNdUS3V1tTyqqZaa6kd6XV1do9dfjBkTGVeYvb7+jbS1tklba6u0tbVJW3ubndva9d6MmTN6xb8UPkZFDYsaZNbkDGcz1onr8L0Z6oAnRh/6/hv4p/BT+qf858Y8lb9U/yR0brA/rmPNHuEEp/pXA9Ve7M8nEYFCEZummYisktGVGi1P0SYMeBTVWopblRNGjgI9LZoLkVowiG4ozZAFg+gTlMIPmQtn2EC7yGlI6e/8EzlWKf+pgrPMR3COUvlL9U+qf1P700/7+0lktKlwV2PjhpmzH7FDQCWmKWAzUBl1FMJ/kk5DMFz6TB1MsvLfMgxkDFL47iSl9E/5j+g2lT9z9Fz3FOuWVP+k+je1P3Fw+O/a3098nVsNsa5j+8PdIFl0j7FOFr2ZkMbpOJ2UUgcgFOZFRp7P/CaxhS+Fbwoupb85mbFwp/xnDlEqf8hGqn9S/ZvaH7fNiXNyuflv2F8tAiwy3kTrRO/hsO9iLzz+Lijo6LfxZ9am/P/2e9LaGV0z9/vJcwo/zrDo0klE07Dc4us3RXMT09toGX9O6Z/yn8tXKn/wQqp/Uv2b2p+kXXD9EC8BYHSKvAlPwZkyjSMzM0qxIxAr26huQKP8ONKfMOFLOfPrr5FT4FF/qbFLwj9ZWSmTJ8e7DPqCz9a+aVOnGYwS+FrMlzCq/YH/zZIlsnv3v6Ix9wV//Pjx8uuvv0pFRYXVTxRlOorTduXgr1q1Sk6eOimnTtrx865dYQmm//RneyZjGP7Z8DCGnORyOVm+fIWsWL5cli9fLrNmzoyeC47/2r07wjFJfxRGNM6QyXGmgRa9zf+sWbPk8KHDEXx3TpL0i56byco0mjudqLQx/D/iv78z/38Xf8X3H+C//0n4E+Dl06eloiJ2KpPzVzr/Y8eOVb4bNHBQv+Rv/IRxJis5y244X/0T/Ndf+i9dulSQsSI+/gj/ffvNN/Lzzz8rfqX4R+Nn3tEzYf7XfL9WNm/eHOo0EnrSC6eLHPq+9e/27dvl+++/t51EQZ+hb+jRokdF4sx1xUDTRxRT/0X91xv/sf1744YNxXQLzx4/brxA1+9Xfy+zZ82WoZ8OM3r8g/Ch78EDB2T2nDnh2Qm6ltDfM8179+4VdF00T3+T/v7//s6/w++P/mGMCxct1DHy/MGDBsngwRyD7Rg0SAbpvcEyZPBgGRTu+5j03Av+H4P/SbzmaIwbCwTGIyufDh2q2+XoBkjjHLbhfTX9KyNmADhkyBBZtmyZDBw40BCICvrMWVi5apU+I5erCP/LyQ8bNmhjn63btth561bZvGlzxFiFQo9s3Bgz2thx42TipIkyedJk3X44cdIkmThxkho4kGd8Jmxsu4vT6sWTFRRaJifHjh+X+/fvy/0H9+zM9b37cu/+fRk1apRcuHhJWlqao/EYXTLaJfHVy1fy6tUrefH8heGTzSr+jIFJ6g987VSI4ISxHj16VF6+eCkvX9px/cZNpb/Ph8On4AsavGtokOHDR0TwocG0adOUDlOmmOPEf2AWujrSJbGzs0s7Mipj5nK61ZIdHRETFSkkm/9S+D+s/0GOHT0mx44di8/HjkWNoo4cOSJdXR36TP4L/eEhGkJduHBRVn//vS0BhZqTHTu3R82d3CH8+uuvbcvmqpWygm2bK/28QlatXCWzZs6yMQf+Y0tqZWVldJw8eUKvT1RWCo7kT+pMoSji+S+lvwsQY0aZfjZ8hDa6+mz4cO14yfmz4Z/JiOEjzMFi7lzgslltNtXe3i4cHR1tembHSUd7u7S1t8vhw4fLwv/q6+nKt3TCBDfg2HwU0x96V1aekJOVJ6XyRMDVz5WVSiOfx/7w38NHYfdMje+iqZbd/9qt/I4sI38oIOiBkZk5a5bMmjVTFTv0x9GbGhzuRYsWKd95Z9ARIz7T3UB79+2V/fv2yd59+/Tz0qXfKW7LvlsuhZ4elRXn/yT9y+FvfEgdUU7u3L6j8Ap5dFLBeo8kDBz445TMXzC/yFgm5//K1avS1NQUyd/kyZN1i/OWrVuDPtom27Ztlelfma5jq3JjU6PCj+Y9k5Vhw4Ypb589e1bWrFlTJP/QuL6uLtIhSfi98R87neCXtvYO7WJ69NixSJbq39TLw98exnyXyWpvFdPNRguue3qseyt9Ttj9ZLTrH/8PHTZMoM2fjU3S0PBOqs6dUwPk9H9O/5YnT4vkD6eEHVbApnMs8q/XhYK8fPFCPh89OuJpm2c62gVZjGpeTJ5GjBxpuuXYcTl+/JiAP7z/3bJlGpAw/12dXXLw0KGYDtmsLFy4UJgD5GzcuC8imqF/Wtva9Dun/6iRn8us2bOVh2fNnmln+HvmrMDns2TgQPR4HOg4/sBPzr/LXPL8MfnD7i1Z/M0H+KM3kG2fq3yeeSxorxpoif5uaW1R++t8z3no0KGBtrEj9DH4pfT/JGppizLDA04oNTOeOVm0cKEg5BwLFy2SAWrovZAvpwRkkHQMBIAiwbOCR71q1UplCKJRI1RGGbepqVEaG5tVEDs7OrW7oMPneeZpmvfa3Nysz3DGwhmht8CnQz7VZ3LfHYBS+FHEGSmJjGBw6U9w5849fS5bFe/euyt379yVkSNHysULF6WlGQeg2Hs+fOSI/HrmjJw586v8euZXUwjuAPSYA9Af+NBh79490p03oaFtMt0V9ZzPqyCxjRJGOFd1rmiSl3zzjTpj06ZPi+DzPHcAJk2erLS/cPGC9nGgDwLdGPV818548vRdaGpqsWd/ZP7hA5u3rNDtsa6+Xurr6wSF1N7aHrdgzmZVADs6u2zuc1kZMWKEtLa2aFtp/sM8MQ5/3vbtO80BSMB/+vSp3uvsQAl2CrxBQycOFEzD+/eWnQi7Uy5dviyvX7+ODppOva6zA8UEfIdnZ3MQuS7H/ydOnNBx0r0SQ4jTy7j9QLGOHDlCcVSFkM1qdgVHJXJW6DuhvSdWajts6BaPIaMy09j4p+QLPYpTa0ur8jPzf+fOHZ1Xlz/OL168sGZaNNRSXGsVx7raOu2giTPrz+8P/924cUNu3Lgu16/fFK5RJvsP7Ff5NwegoFEI/P/FmC8Ud2jP0a3nbm0LDv7oBGjjDsAXY8dKw/sGOxoa1JDw/enTp3WMy5Z9p7/HWS5Hf8cjib/f40wwQI+RRYsXyeKFi4VgIPk9+NMyvKPTHNFy8n/lyhVB/zj8LT9ukabmJmlqbFJ9xJn5p2EY46BvSSMOg+u3bEZmzpoZyWtDQ4POH06FGrxcVh5VP5K6urqiTJqNs3f+I4u3du1aWbtmraxbu1amTpseRe/1b97Ib7/9FoyE6d8ZX88QsofJg4Br1epVul2aLqjl8Hd6Of6qd3NZaXjXoLJ65fIVuXrligYNb+rrI/rSqO3JUxwAg1+RrVD+owHc6GDooX+uIifz5s9Tma2ufhT9vy/8yeLSpRV9DNy6+jcqf+ZAmx4ioDl06GAkfzgD8BdzgLx3dedl1uxZpoOyWWlraw0OgP3/2LGjysfGyx/qX561aOGi4DwW6//k/GPfXP6dntH3CfuXpD/0/e23BxE9nP7tbe2aAXb7RzC9b+9e1T+XL12Ksjw40+if+/fvycBBA/8y/FL6WyOgYKj5kmIbzocPH5K6utfKwHp+XWfX4Yxne1rT+jklNgbZHICYuQ1YVpkTouIAKHMATx2N8NtcVrZt8y5/do/nbdjwQyCUNTTR5wVDcfzYMY1s1avLkQHIBwfgQ/g+Ds6l8MeMGaOCi9Lle8cf49je3qatgidNmihLFi+xBkQ0H7p+XW5cvy7XQgOiHzb8ILFSsyxIEmbyOgl/yJDBMmnSJG1KhNFr/LNRjVVjY6O8evVSlyD4ftiwobFjlsnKgQMHVDnZOw5ycuPmDamtfaXvXIAORDPAPHL4iFRVVcnVq1d17p48eSrnzp+Xc+eqZOWKleoA4Fnu2LlDaO+cxN+Y1mn5If1NOWfl6VMUwhPFH8Nx69bNIocAxck7Hsiq8Hw8YDzbuXPmqhLZsWOHOX7ByYD+0OLZs6Bk9H4M/8b1G9px0uH7/MdC5mM2YT937pxG5z4HSfpH/ynD/73hb9ErjueQoIi9P7fDzciUKVMiZxi40Jgohmvg00GTKIY5/ooIM8An3bdp4wZVTlevXkko7hh/V9RJ/N+8faMKwXD0ccROm+Pu8LWYKiF/S79bqgqU7Ab0Nwcgrw4y8zdmzGh1ELRtdpC/JHwMMXxn2YsP4WMYkP8ZM2Yo/suXL9PnDRoUZKUf9B8yeMgH+ud1XZ3UoY/q0U04fHUydqxFf2RJcBiTuCfxx8DhAOj3vcDHcYR/4AUCAs0YJPDHEcZxGzSYpY+MEL3SCO38+fP6XLqYmgMQz0Vv/IduRE7y3V167tZzuM53abbtDQ7Ag4fB8Bkv0eW0s6tTnj59onLjZzqsIkc4Okka9Aaf3zA/PYW8/PADepd5zOh1oSevWUd+8+wpHVyfJjIrOYVPtoBGbfPmzZUJEyYoD7EUCE44mD6Gj8HnN65/Hb4GNT15WfbdMn0m8kegdPjQIeUlImA+nz9vQdKAARWC/qTdu8PEKSA7wOe+4E+a+KXy6sKFdIh1Xv64/Jn8Jn6bCJh8DH7GMX70yLM44T+5rGYMWcZ2/JcsWayBA7jkC93y086fNLjo7OiyNvb5vAaigwYNDvqjf/BL8ddWwJYy8JQMzGpR/aZNm2XTpk16dMGc+bx2ALR7m8WIlNXUIAI+bvx4ZQyUA0TRjlPZrKxetVLyPeYAqMLWVHNIA2mnvIxs2xZSwSqMOZ2EDRs2RpPIwJ2InJ8/f6FCb/cy+vvNP/5YFj5EjQsgLIWj/Q5yObl546amzIi0MbY2mZYeJ/rjhUi8Y4Ao5/Lly3L58iUhG0E6nWvurV27xpRmT49GTaX4fww+62ak7O7ffyAbNmzQqGbtuvUaaQN/pq7ZG+M6/sDv7s6rZ0uEzTsXmJMDBw6qcZ0SMgDQn8ixkO+Whj8b1BCC5+7dezSzQfdFIs7379/J+fMhQg30N1iesSmG79+500OqFGXLC6AwBF28lCnMPwz87OmTiP4sAwHzLFmNXE52bN8hhUJ3nB7PmVNBpKG8Eubd55+olZbT9p0xPffwrBsa3sv7hgY9ELR3796p8W1rbYng98Z/Eaw+8OedE/B6En6sKCxawDmu+b1G+Z/lBJQqTbOcbosXL9HIf8KXX6pjVyp/N2/dVh6LxpTNqnEjwkHR2vl9dI3RYfmK38N7f4X/JoyfoCnbx4+f2PhyLGctVxwvXbokZENwkrVtdtGLs+ANoz/RODRhSaocfJp1tXW0RfijzMl8NLc0qwEvxT/GO+a/XEVFpIvg9Tt37iqvPX78WDZt2mjfbd4sw4YO1TGQASCtCs3Lyd/lK1c14seh6g0+0bM5AFlhCUAdgEgP5VT+bt68GdGBcb97+1aew7u5rC634QCUg+/6pxT+sM+GqS4kY3fq1GmZOxdH2eRPHYDffovoCP1ZiuSFa7RZr619be884fp1rX7H/UEDB/SL/3Fcyaqu/2FDBINMIXPrSyHaql0zAPH8T5gwXjOn7e2k/lkmJnWdV0NMTdKAily/4Kt8lMgf9EOHXLx0SWUZZ5rxHKLOKJvRF8wBk7E7/589c0bhu/1pb2uVM1XmAChvlbE/znO8Bwe9izPk95z+Or7E/Dv/x+e+5Y+M5uvXLAsROM/WQGzVqhVqT3AA4JtZs2YrjgRX6A9kkPb46Fiy1izJrVu3Xud93rx5RfxXTv4+xn+aAXDlqghGWwHdo8jKsaMhZaKphzjVqIomm5OZs2fpgMeOHa+tWG1vYmywV69crd+/VyX9TjMBpHeZSA6Ui57zeVOIuqaPYXqva+IzdTLiJQVPdW/etEknCePO/588eaYthad7jUKYrFzYdqhbEMO9irAGzv+2bd+uTAaB582dpwqEDACGFpoYnjF8Oh3iGDj+wCdq4g2J165fkytXrgYBMhqUg+/PxIPFcCLIRIkYxF+OHJGammrFibUt/y3KAYHGESPtiaFtbGyScRPGK7ypU2FeHBnLAOCJ83nHth2GR4bujq/VgPBMojtw7Gv+Hb45R4YTxT757m65evVaJNyksw4fOaxpexcWBPb40eMRfO5Dv0cPH+o9MgCklP33KMRnUQYg0M+FLpfVaMIcAJ+XnNS/faPPPHjwsBw8dFAOHjwkvJXSj21bthXBL5rToFyT8A3fmP+T+FOIBU3jZ8TKDfxpRYyA8yItfsP8Xjh/UdOjzn8sMaEkqYvgN0n6V+Qq1IEhO+BjAj5zfv/B/QgncCQKQhEePHxIi68cfqn89cZ/SvvublUuWrwKnTNZfREYSpUUPWOguyY4L9CoKJZro0tOFmoGgBTsn/Ku4Z1oVBLmjLR/obuga7SOP7IC/nQNJWpM4q8GUf9bnv6MB4ektbVN+Q9HeNac2UbHQH/wrzxxIuLDcviT3oamOFPU/ezfv19xZCmEsYE/OOMAgCeygp6iRseXcx7/8Ycq7klTTN6WLPlW/3P6tBU8P3pUrRmAcvAjnkrMP848y13INfLP2BjDj1pImBUyPaTa0X8jR43ScdHefPfePUINye7de4WXquHg89nu7RbS9NDN6a/XLlMJ+NwnC0ndyoED+5UmRM8shfId8h87AFkZN3asECRu3GxB4saNm2Xjpo2ycaN99uBx88bNuqzRH/g+/2TJyKCgZ7+cMCGCzzi4B/9DQ/i/s7uraP6//XapziH05Pdtre2aASgHH0O//8ABez61BAsW6n9xahTnMvq/+P6H8t+b/H0+0t5Ay/gHVFQIzivXFjgVrO19sL/z52PYY/1369YtlVO/x9nlz4Pbvyr/PCN+F4BOsFflmvBRvU8kg7e+6+dd+g4A1qXxyKge9xQeBUF4juNDDUDE3OrNZaIlAIqyThw/oYyEYThXVaXpIgwVBxkEUwYZLRIiKmEde+Kkyfb2P8aYsfqBFta29PlGJOD/+b5BBXShRyoBvhNNn53JysBBgxQHBB3viu9Zv0TAEPIZM+xdBU3NLQoDb03fvpbNyqCBA3VdHnizWWcKk7RsuSk1Im6NAJSejC1WZA7fFZ7TCfqxBMG6NevbrJU/ePBAtGhKC1FyMnr059LO+lZnp9ZigDtFkUQpMBHrQdOnTVWFoRkAGKQip4VEREIwUHVNjaZer2h6OSuXNKoJwl1m/tUDDvAd/8mTJsnt27d1vlFSjj/jwdPUIkB/LbOmtto0S8LvHH/WWnkG+OsSgDsAYYmINCbrfChQ1lEfcn70UKofPVJlpA4Acxvmn8gI4wh84Dhd7fu+6R/jUMz/5fDH6HZ1dRvsXFYVG0qSgj8iDa6tdiMf2k6HFtRtbVa7EAotiRzhPxw/PH8KGW/dviXwNRE9jl8SPve2b9um/K/jTeDvn/uL/792/0ujCcZ58+atqJCWjAxzpJkdLdKz9OJorQHIi8qVb0lNwNciwJ6CnD51SmtrBg4YoBHUq9pXakgxBIzR55+CLgwblc4R7fvJf7t++kmNbnNLiwwfMVyNJIb83v178nUo2GP+kWurASg//6TGkSV+Bw+yrIYjOl510XghO4P+w+lG/tFD/P7C+fNqWOGtUZ+P0qwAuJCGZz5Zv0ZHgJfVALyOcHT89Zzk06BDNm7apPonNj45zVL88fiJ6p+3b94obxGh3rp9W3hXC3pUjy5zHLhmnH6P7ADXvvyCge0NPvOPY/HgwX3NnsDnLO99MXZcpP9wzolMwX/turX6O/QLr4nvQAZCIazfYxmVa2xGX/gDn0zP9u3bVK/h3C35ZonS3/UP+p9xaRFgLqs7M5j/AQMHRPLPu2qYE1+a8SWAcvD37NmryxQmQzlZs3at8uwQrS0jSw29Yv1vuyOC7gn6x/5r+o/r3vQPWaTurm7lM5yz6HfZsARw6pTS6JtvloQi2v2yb78V0ZLRxBGjLgDHWQtr9+9T29pf+OXwL3IAXIFQ1U8RDESkUInCph3btiuxSVvQhx8PmYnAcM8kZZFnCcCKAHVAgalRYlERYIUXVNi60dFjRyPhMMLFwsrzSIlHBlTXLLNy99499dznzp0j7mkxQfyeF/wUw8YAx6kqJtOInpPzF87L3HmWXvP/gD/KjM+s+VkGIIwpwD/162mFT3UtkZ7VNeR0jQqnwKMmf2Zv8AcPGWxFXKxjhqM+qrkIa5t63+owpk6ZqoV8X06cqOPTCtWMVaxDfwSbokDmzGsAGMOwYZ/J1q1b9L+k/HkZkhsW6EV9gI/V518/B3yT9Kf4iqUchPm775ZG9P9xy1ZZ+u1SFRaKJFE4/kzexfD27Rv7nMtqihYFpnOfK3YAHP6uXbu00hun6M6d2+ocUfnN59t37qiRcfyBQ3EUa+oN79/p7ghNkZNtCssCpM21PiUx//CCHhGfxgLcG/7cp+CQVLDDJxrdtPlH3cECPSlELT5+lM2bN+m9NWvXRPwHbGovMBhknlgrbWlu0iIvIjzm02ho/IexRpGx+wP8WOp4p/g16D2tzE4qpAivD/l/0cLFcu78ORkxckRUXAQsp/+cOXN0twvpR+YfHDFu1ABExkOXeGyMCxctVvnDIPMc6k7gQ9bn5+vbNoMMBfrrEgOyomvnjmcMX/EuwZ96Fpzz7kJeeXb8+HFy69ZtHRs8xztLkL/rYb1ZHQCtAfgQf/gfJxhH1GhsdTXdXTHfltLfawDK4U+h3s+7ftblO38eZ6sBoICuGH+Xv1L+8/Vnsnz79u6Ta9eu6fKRZooyWcHRffDQlgBY616wwIqzoT9Fa9RieLF2fF6sS7Wql/rJ/9OnT9cdKYZLsf4jW0rWxvmfzNHKVav19xQeJosR7Xq1TJs2NZJ/17+9yR9LHxj0y5cuhwr3jLDFmiw0/Aj9CXi0BiCTFd4sS4Dq+ocxE1zwGxu/LdVoDUAZ/Nl+h+zxW/ifLKz+t4T/nB/KzT/2R7//iPzhgLEswq6YSxcv6/XyFcsj/sNJOnXStkPDu+9ZznzP0WDn8JkAkft/vn+vS57o7/7AV1qUwb+4CNC9w1xW05NUmzsRa2p+Fzx6IxSGZagW7kGQ2SEDcOjwYdm/b7961Qzs+fPn6kCsXrVahRemca+HCO/0yVO6XQeFR8Szfdt2LSwcO26sKhAKxhz+6NFjtPgERbRpI7UBxpg+IQj/JlJl0cQlFYt7vXE7Yv5HSkhfTfzc1ldgPNIxKAYK/eZ7JiFsH6OqG/hEOnhpKLm7d+/pmgwZAD5He6EjBWw42zhj+GQhTlWGVxSfil9VfLaqSqtBiQorw/1Tp07pzgRXGCOGD9elDgwxdMTjZ8w4RxTe4MAZ3bLCljqcI6/c9uiUz4XuvKbO/bnQTscZGNlpa06MGUxqEmwOY/qThaAwiN/v3bNXWlriqnt6KVBF79t4PPWqKcxMcAAKYQmgH/Cd/5Lzz7o8DgU7O86cOaPzQG0GnznYtjhc04Ex/RlrlBJ0we0HfCIgtjYl4SutNVKwFDpr5xhpnwPoZbT8+/Dx+sERRcjyAhkBcPtF8TsqbLlTGH+R/7Fn+fAAABMJSURBVNk9QZSnY81YhT2yFEWMLAF8wS6AHtm+Y7saGJxItuiSNmcJaDXbfHsKokWAmawqbNL8ZDVQ/qX463IZGQDvG9AP+mOUiJooJuV5FPshbwQkTv/FS5boNkU+o0RR5Lt2/aTGgteB40y+rq3VAObylSvS3BgcgJw5APweviSzBv5sWeV16Dt3/qRLAF4DUMr/Dh/6Dxs6TFP0rN2bA/D6A/z5fW/89+3SpTof4EaWg94Dvn0aB+Dhb78VyR91L+gk6M9/Sg+c8VL6fww+46o6V6URuPKT6jGcKJP/p0+eyIvnzyP+Zz86jikZALYvUs2u0X/IBDCee3fv9pv/oT/638fImQwN/KeOMdv6WlqFLJb/BnjYpooBA3SLOPNIsKBjJrpua5MzWgT4ofypA5A3BwD9x3LA4yePo2fbM2L8+fyx+S9nf6ZOmaK0wZjzX5wmHHn0L8E1+hcc0PlK81xW9T06E1mhfwtOMzZGP+v979RBiuYooX8cbz9/bP5DBsBTn4aoIegGlH79OV2TIl2pxiJDJG+eNUadNXcYjUimrY10z59ahEJhGx4V+2OZQNsGaC0tSU1xzxi2Rw0URIDJ58y2IgjPALAEgMIDBkaYfexJ+BgonmPbAG1cZtTCxCWZOGppzFas2ept8tZDogga5uDdsu0CZ4A1GhQ9+LMrgnt3796N4FN8RCbkxYuXVgSofQAGm3CoUvs4fMbNPuJ79+6rIXd6INAwMZkWKrDVw8PhQbnnspqdIeKlToHofvPmHzXqwvBSnKhLMwE+NKe4kboAO0/UBkvsbMB5sSxH3/OvBi/AT9Kf5RLGjRdt28CK6Q+zI1D8BvpBrz269o0DkQtLAIWgIIxe5fivN/jwX+SgqGEYpzBm67pw3/R3ATdB6hv+nLlzZP489pcbnqXwGUtzc5MqZK/FcBh29pauhr9/11/4zP+e3Xu0kBKalIOvtCqhS0wj/4/BR+bI5Dl8FBL8RxbGx+tLAMwfUUxXZ7euVSPnL1++EIp1kb/S+YcXffdDEj7Fw6Sv2Wv9d/CH/8aNtXmeU3aeM5o6BT46CQcVQ8nyHMsspJivXCGT48tfOTlw8GCkj8AfXJFB5pKCQd0yi8MAXTM57cWADqNan8Ja420MMDotrz0DrAbAlwCcF4vp7/iztXratOlazMZSBvRkN8HOnTvkROVJYevam/o38hsZgIT+mz59msyZM1fmzJkTDq5nC3x6+TLb+Dyr8XH4Pv+Mh6VZ8NHteOywqH+jy5JszevOd8vzZy+K+H/VqtWy5vs1qufXrFkra9Z8L2S7uIZ+qjNDEBXzbP/5/9nzF0pTMktO/6T8sVSqvKn6pUftD9kt5zlbAqiK+NlkxuBbBiAf8T/496Z/yMpAf4KcJHyH43NZKn8EatRqmXzY/A8f/plm8BSnbE4dhJMnT+kYgc9WUPhPl3m6uvWsn7u7VP4IqtatW/tvy3/0LgAbvBkZiqde1b7U9UmMfu2rVxplouT57MfLV6+i7RUgnWSipCfELgCiUNZ3KFSAYEOHfhqleAw2ijzebsHWE3MAzBFhTBhLg2H3ktcInVVaB0ZPREKa2lTjlYvgA4tKYRyLcvDPna+KUkN8T/EXEVcSJtcTJ07UPd2sm5Iejwqg+gEfZv7jj99VSImUUYxjRo8RKju379imnrU5IYYv42CPMcoBpyo5FvCZqo2AejSr4fRniYYqdOhP8xUKFd3RIGpAiZfDv6hBSyKbkYRJvwEULIqCtT4KwCKjV4L/9GnTZcPGjTJiBNsBmSPbR7xjR9wICJ4glYsj09IazlwnDqIi/0w3MOdFiuZe1r7S1z9DH9L//l3tq5dC8ybG6/zn8PUc4ZfRXSGqTAq260UVO4ol7IF3RcP58ZMQOev/M8rfOL18Z1s6W3WJLMl/q1evkZbmVm0yZXg4npybpaU54NfarFkEii2RPw7HB6cNHJE/vwf+f2jk0n/+J6KH/xkv2zfp/IdyV/q9t2LdmDc+lDnnBa0BiLYBxvAxPjgASfz7or/zbZL/KKRM4g/O8BxLAmRajAY2/2yL0908JfxnznMs/+YAkAHAENhSGkp54ADfmmj3HX+KAMkAOM4ss5FxpEqeqIztbxTmRs25MrYEwNbE/uBP0Rd0L+TNUWbrF/RjrlkiJKtEcEQfAB8TY+F79DJbFmkUQ3EkDg9RMme2KvYHvj3T9C9GqTvflcigWeYJ/fdnY6Nmdvk98HHwGTd77clCNje36Jj0NfLh9fGa2i6jf9WYB/on4et1aBCEE4RDxtIAcPbt31+Ev/0vK2ypJmPDsogGmon5pzbn7JmzuuvoypVrGiRBW+o24H+ey7Xe6+iyc1eHBmD+fM58r2PYt0/H4Lxgv+nd/k2ZPEno76A1NsH+wXfUaDn/oT91F4CPO9kgKdJPnkm0ove169ZFtOA5Np5Y/nzsH5v/RA1AzPB0djpz9oycPVslZ8+e0S0Udn1WzladVaPPmkrVmbOa5nMPyD2nYsJkhcYUTKIWeYQJt6g2NmxOCB80kYZtA/QtfAExjWzsf1EKN2QAvFq2FD6fGZudDU/gs+zAhFJZS6pw9uzZuoaFJwZDENkrYZPRVC/wUQLgqM1NEhMGPr3B5zv2yMJYbKWjGBAvkbQ91bQtLUQu70KVfYw/URvKaP/efZqSnTt3ni7ZUCSEMv90qDVH4vlfTvhSFSVpWxq00CWLFBuFPXRL47AOjvH825j5bAxqVaYxfIySFgJ2W5SA0kMhkmqDnrfv3I7SdX3hHxcBZnSNj88cGKcdO3YKDsJ2vcf1Dl03999QVQsfOm9W6fVZTffZ/bNSdbZKP/OdLjv0wX+sN2LQaMu5iLXVRQt1C6hdF6+zTgm7LYjcUI7gjzKG35lHPH8cL/pFLJg/T/lg/PixAb+AT8AXPB0vx5/MDhEeW5gcx1L5S+JPbwynN2c/yvHf0WPHda6o0ifqRxEyXviH+SXSY9mPpb7S+ffnuvxBG+adpakkfPia/ei+TKHnY7/oUsYvR49pxOrPKpV/ir6c/wYOHtQv/H3+R44aGeHu4ymVf90GGBr72Bg+zv8XL15SmTOj9aH+4Rml+s/7AJSjf1/6L4m/018dgFADoPAy5gD88cdj3Q73w/r1el63fp2eWRrjoD9BKf4fg19VdU5T+uwwWLlyhazQhlZcr9RCaZZ2Hb45AD1amLZg/nyZP3++NgDi7AddJP8KfPpG7Nm7RzM08OXW7eziyVltSU9BHRsytdESUsgulNLf5tU6AcIbzMPUKZM1i7xCcbPGXeAFnnb2Zl4rZMYMa5sO/Qls4HHsAjg7/gbDdGNv8Psz/5EDEOS23PwrrGB/GMv6te4A/H346gBYMZ1VOipiAMn4eol5FupFBGNGCioWBLzGIDzuvXiEhxBnsrJsue0rtiWAWDEZ8egXwL1i+OwV3xAaUhgBDQbX5eCjuGglHE8Ivw8ZhagD4YewWV95+bJWvUKUHs8hMqNC3qqU+4c/jXRQ9lYEaIazP/CJbqjEZn8o8JlYGuWwHYhtRuPGjS1xXLIyftwErXrHEdAIrrtbI/E/fv9DnQdXnMAf+8VYdQDAq1wUy/3vvqPNZjH9Pzb/De/e65o/hsJwtLkZUDFAIz7SXazF9wd/BLm72/bVG9OzL/uf5T8cT+e1cudy/Pcx/Ev5j/c3YDwPHTwUslo2/+Bx8MBBLdpBeZeDzb1/F35f8leKP62ZydxoxBHgHz58RPkPg8G4adTEdjCWmPqSv/nzLHrVd1CoAjP82bmhjXoSha5Es9rAp64uLNn99+MPzdnKx5z1l/7nqs7q73vTPzq3JfqPbm1knkrpr79NOmhl9F85/oN2bFlM8h8ZxKbGRnVOoi6GTXRX5V6jLhfSw6QUZvJzKf9R90CRMw4tkT2tdH2HC9e3bt6O9D9ZI2qQND3dScqag5S1pa9ZqiQFn4RXel0Kf8PGDQobuSIrk+Q/nIk/fn8srW2tUUq9L/4nS0I/glK4/rkUfjn9c/DAfq1lqiRN/xftX3/mn6Xz4yfCdul+6D+yt2Sfkvo3yp6V2N+PwY8zAIkCPTUgatRNcSrTRykGjGgwijAug/VDvWA3mChxN7jBQVDE8JTD54QQ8Az3oP6T4Zt3aW8vM0PuaR+jqzFtSn9VCin/xbKHbP0F+fvAGc9lZUyyZ3sq/+oUQ9NU/8U6/z9V/+DMESxZgfX/vP4lmzmI5Sq3vX9R/m0etQ+AGXmMMqkO87bMqPNZix08wo+MtBlwNdhq5C1bwH9NWGJlZEopvh9/71mEFL7RKKV/yn+p/KX6JwRC6OJU//6vsT9kdqdMnRKC1P9/7N8npDvMKCfecuSRuRv+pGGPonqP7sM5qkzlsz/LjLt6G1r84JF/OGsbzhR+Sv8kzyT4KuW/KG0ZOc6p/EU08SjG9YtVZid5KdU/EY1S/WvpcrVtqf0xmUl2AgzK1tPwrnD0jJMQ0q32vUftdva1W12PCOl9f44CcifjA+XlDoGt+fJb/18K3wxhSv+QjUr5T9N9qfwFA6+6ItU/6MxU/7od4WzXbkdS+xP0J45PGftbVAOgxPLoPawp6D1/PwD3oocgfP5KYDNWeOBuuJ3w9sysFixE2/pouJJj2cAEuPi3Zboq/QPwKdLz3uvg0B/4tAedM2eO4qwM9W/g73TQcz/hR//5B/CPnhUYoT/4R/9J4Ztjms7/35b/iJdS/lMapvL336v/U/4zG610SNif0AcgTAZfaIokGHTNCpiRplr7wP4DVnTgkXpwBuiZTZ95esPTpYz3B/AGvZ07dpri5AUvz57K02f0tMaJsGfS1pJmNxw11TVSrb3f7TP3NmknwISTwH+DY6IGOeGMbN22TV68eB4r6mxWHtx/oC8FweDry31awj7eAD83oEL27durb1ii1Sh7gJP4d+t7p3nrlNGDbmTVNT7eah2vjb9GHikej2z7WxgjcONx4iyFQr4AnzU+z5pYrUXv9Pcx9Ia/jdtqMfw3KfyU/in/edCSyl+qf1L9m7S/2B/LAGCQ3GgFY5c0hHirdNH6k/dnR0Y39ijoFU7r1Sv6ulxemXtZt4bQAYqGNTTRoXexvkQiPB/FRL9nWgZfvXJNrl67atd8vnpVeP0we5QxfMC3swtzDNsNI60/ecOgf+ZMz2T2q3PNtj6aVPj3wGeLEpWdT5880y1Qyfat4M9WFhoQOfyJkyfJ1SvF49TxX70q/nIX9r46DK/i57Mq4kTGI1bMwSHqg/48Qx2GMvR3eAYjSZvYeUrhQ5cSejgt+8H/Kf1T/kvlr3f9G+u4VP+4Pv5A3/wv1D/WByAY5XgSQyQZ7tMohr7FdJDD2JsxCesuWvwXrvl9UKoN797Knbt3tSfzi5cvtdnNE32PtDGIRb/hugx89rjTB9uJaU1HTIGXg48DwB56GuQ0NdNRrUmNOw4AxpxOXvTNp8Up+9fXrrf3XPMuAL6fQc/8QkF+2rlTo3JguAOgdFG8eodPQx3+P/rz0Trm5FZHd6b8zPP6wt9wjIWpL/ydTk7/FH7Mk053P6f0T/kvlb9YtyTlwnVrqn9i+vRH//9f1b+6BECBn094JtE0hXukph8/eaptJmkDSZRM8xpnGs5fff21dkCjC5ofGMN7d+9FRRm8R/rZs+LWqUa0jBYYlsInMuclI1ZX4K9ljCclCR9PSzMAnZ1Cmp6DN+PRVvX27VtqkHEACoVubSF69swZvc9YScE7/jRHeVgdv+K2W986ddh68XshY3BWSuHTMQ6cnRH43mocYkOUbG/qxSpJ+Py3HP21a1cf8B2un1P4Kf1T/sNhT+XPdUKqfxK8EPFFrP//E/WvbgM04xuIo5GuXU+aPEnq39Rpenze3HkyaeJk4QUYdH6ida0zlr4NsNCjLzBYuHCR9rSnze74CePl6dOn0tzSrC/M4BrDRLMbeiPTkYxWo8OGDo3eDcB93iCGMV2wcIEM+dTeSR5HwJad0G5NwRgz/hMnKrW7maVdbPzv/wxLADle73tJe1S74eb1qdqnO9HbACeFbn7+Dm26LR06fFCXR2i6MPTTYTpOxsu4cYQ4+EwXOByKoUOH2ct4GJun9fVs2QOHT6SuRjoBX+mZoL/+n+UPfhNlIMrj75F/Ev8UfnAYU/pHyx8p/wWeSOUv1T8Eaf/h+veT5DoFBghjOmTwEOENeRhhIn76o2OAiE7ps01anfatfMd71nkffU9P3t65juHDWOVyMnjQYFm/fp3wDmnW43EAgMGLbuibz/OjF9OEz7TB5U1HvOiD78kE6LgCfDdq9lIRE2aUWmWl9Tbn97QR1v/nCxrp8z0ZAFpCqpHNZaSmujp8NsPMffB59+6dvsXq+zVrFPbhw1YE+ODefcn39ESv3Uy+VIf3UYO/4VOQ2le11v4zGPNI6QaHBU/cMx7l6G9Zjzjb0R/8c8mXRwT6m1MQCl8ctjomKfyU/s5fMf8rn4XMVSnPuvz3Jn8p/yWiy1T+Uv1HJv3/gP7/L2DNkY2lCuU0AAAAAElFTkSuQmCC)

# ### 하지만 이런 것들은 정답이 될 수도 있을 것 같다. '똥싸다만것같은' 문장을 잡아줄 수 있는 방법이 필요하다.

# # 1위, 2위 빼고는 3개씩 슬라이싱해서 보여주고, 그걸로 안나오면 하나씩 보여주기: 

# In[ ]:


# answer = OrderedDict()
# for num in tqdm(range(len(test_dataset))):
#     id = test_dataset['id'][num]
#     query = test_dataset['question'][num]
#     tokenized_query = tokenizer(query)
#     bm25_docs = bm25_ori.get_top_n(tokenized_query, wiki_data['text'].to_list()[:56737]+dummy_data, n=30)
#     bm25_docs = [v for v in bm25_docs if v]


#     for index in range(7):
#         index -= 2 # 처음에는 1위만, 두번재는 1위 2위 , 세번째는 1,2,3
#         doc_accum = ' \n'.join(bm25_docs[max(0,index) : min(index+3, len(bm25_docs))])

#         if doc_accum == '':
#             pass

#         ans = mrc(query,doc_accum)[0]
#         if (len(ans) <= 1): # 기존에는 if else에서 else쪽으로 로 처리했는데, 여기는 그럴 수 없음
#             continue
        
#         if ('다.' in ans) or ('unk' in ans): 
#             continue

#         if ans[-1] == '의': # '가', '이' 
#             wsd_result = wsd(ans)[-1]
#             if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의'):
#                 ans = ans[:-1]
#                 if (len(ans) <= 1): 
#                     continue

#         print(ans)
#         answer[id] = ans
#         break

#     if not id in answer.keys():
#         print('3 슬라이싱 실패 하나씩 넣기:::::::::::::::::')
#         for doc in bm25_docs:

#             ans = mrc(query,doc)[0]
#             if (len(ans) <= 1): # 기존에는 if else에서 else쪽으로 로 처리했는데, 여기는 그럴 수 없음
#                 continue
            
#             if ('다.' in ans) or ('unk' in ans): 
#                 continue

#             if ans[-1] == '의': # '가', '이' 
#                 wsd_result = wsd(ans)[-1]
#                 if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의'):
#                     ans = ans[:-1]
#                     if (len(ans) <= 1): 
#                         continue

#             print(ans)
#             answer[id] = ans
#             break

#     if not id in answer.keys():
#         print('하나씩 넣는 것도 실패:::::::::::::::::::::::::;')


    

# with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/predictions_0506_doc_슬라이싱3_이후단일.json', 'w') as f:
#     json.dump(answer, f)


# # score 분석: 1위와 극단적으로 차이나는 경우

# In[ ]:


bm25_ori.get_scores(tokenized_query)


# In[ ]:


import pandas as pd
df_tmp = pd.DataFrame(bm25_ori.get_scores(tokenized_query))


# In[ ]:


for num in range(50):
    query = test_dataset['question'][num]
    df_tmp[num] = sorted(bm25_ori.get_scores(tokenizer(query)))


# In[ ]:


df_tmp.describe() # 통계값


# In[ ]:


df_tmp.tail() # 67276이 최대값


# In[ ]:


df_tmp.tail(1) * 0.95


# ## 결론: 10개를 뽑아서, max값 * 0.95 를 넘는 값들에서만 MRC 

# # 여러 출력을 내주는 pororo

# In[ ]:


from typing import Optional, Tuple
from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase


# In[ ]:


class PororoMrcFactory(PororoFactoryBase):
    """
    Conduct machine reading comprehesion with query and its corresponding context

    Korean (`brainbert.base.ko.korquad`)

        - dataset: KorQuAD 1.0 (Lim et al. 2019)
        - metric: EM (84.33), F1 (93.31)

    Args:
        query: (str) query string used as query
        context: (str) context string used as context

    Returns:
        Tuple[str, Tuple[int, int]]: predicted answer span and its indices

    Examples:
        >>> mrc = Pororo(task="mrc", lang="ko")
        >>> mrc(
        >>>    "카카오브레인이 공개한 것은?",
        >>>    "카카오 인공지능(AI) 연구개발 자회사 카카오브레인이 AI 솔루션을 첫 상품화했다. 카카오는 카카오브레인 '포즈(pose·자세분석) API'를 유료 공개한다고 24일 밝혔다. 카카오브레인이 AI 기술을 유료 API를 공개하는 것은 처음이다. 공개하자마자 외부 문의가 쇄도한다. 포즈는 AI 비전(VISION, 영상·화면분석) 분야 중 하나다. 카카오브레인 포즈 API는 이미지나 영상을 분석해 사람 자세를 추출하는 기능을 제공한다."
        >>> )
        ('포즈(pose·자세분석) API', (33, 44))
        >>> # when mecab doesn't work well for postprocess, you can set `postprocess` option as `False`
        >>> mrc("카카오브레인이 공개한 라이브러리 이름은?", "카카오브레인은 자연어 처리와 음성 관련 태스크를 쉽게 수행할 수 있도록 도와 주는 라이브러리 pororo를 공개하였습니다.", postprocess=False)
        ('pororo', (30, 34))

    """

    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]


    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.korquad"]}


    def load(self, device: str):
        """
        Load user-selected task-specific model

        Args:
            device (str): device information

        Returns:
            object: User-selected task-specific model

        """
        if "brainbert" in self.config.n_model:
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            #from pororo.models.brainbert import BrainRobertaModel
            from pororo.utils import postprocess_span

            model = (My_BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            tagger = mecab.MeCab()

            return PororoBertMrc(model, tagger, postprocess_span, self.config)


# In[ ]:


# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import Dict, Tuple, Union

import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer


class My_BrainRobertaModel(RobertaModel):
    """
    Helper class to load pre-trained models easily. And when you call load_hub_model,
    you can use brainbert models as same as RobertaHubInterface of fairseq.
    Methods
    -------
    load_model(log_name: str): Load RobertaModel

    """

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        """
        Load pre-trained model as RobertaHubInterface.
        :param model_name: model name from available_models
        :return: pre-trained model
        """
        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        tok_path = download_or_load(f"tokenizers/bpe32k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )


class BrainRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json",
            merges_filename=f"{tok_path}/merges.txt",
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(s, add_special_tokens=False) +
                             " </s>" if add_special_tokens else "")
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def decode(
        self,
        tokens: torch.LongTensor,
        skip_special_tokens: bool = True,
        remove_bpe: bool = True,
    ) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos(
        ) and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [
                np.array(
                    [c
                     for c in s
                     if c != self.task.source_dictionary.eos()])
                for s in sentences
            ]

        sentences = [
            " ".join([self.task.source_dictionary.symbols[c]
                      for c in s])
            for s in sentences
        ]

        if remove_bpe:
            sentences = [
                s.replace(" ", "").replace("▁", " ").strip() for s in sentences
            ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:
        """
        Predict span from context using a fine-tuned span prediction model.

        :returns answer
            str

        >>> from brain_bert import BrainRobertaModel
        >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korquad')
        >>> model.predict_span(
        ...    'BrainBert는 어떤 언어를 배운 모델인가?',
        ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
        ...    )
        한국어

        """

        max_length = self.task.max_positions()
        tokens = self.encode(
            question,
            context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )[:max_length]
        with torch.no_grad():
            logits = self.predict(
                "span_prediction_head",
                tokens,
                return_logits=True,
            ).squeeze()  # T x 2
            # first predict start position,
            # then predict end position among the remaining logits

            results = []
            # log_list.append(logits) # 디버깅용
            starts = logits[:,0].argsort(descending = True)[:10].tolist()

            for start in starts:
                mask = (torch.arange(
                    logits.size(0), dtype=torch.long, device=self.device) >= start)
                ends = (mask * logits[:, 1]).argsort(descending = True)[:10].tolist()
                for end in ends:
                    answer_tokens = tokens[start:end + 1]
                    
                    answer = ""
                    if len(answer_tokens) >= 1:
                        decoded = self.decode(answer_tokens)
                        if isinstance(decoded, str):
                            answer = decoded

                    score = (logits[:,0][start] + logits[:,0][end]).item()
                    results.append((answer, (start, end + 1),score ))
            
        return results


    # @torch.no_grad()
    # def predict_tags(
    #     self,
    #     sentence: str,
    #     add_special_tokens: bool = True,
    #     no_separator: bool = False,
    # ):
    #     tokens = self.encode(
    #         sentence,
    #         add_special_tokens=add_special_tokens,
    #         no_separator=no_separator,
    #     )

    #     label_fn = lambda label: self.task.label_dictionary.string([label])

    #     # Get first batch and ignore <s> & </s> tokens
    #     preds = (self.predict(
    #         "sequence_tagging_head",
    #         tokens,
    #     )[0, 1:-1, :].argmax(dim=1).cpu().numpy())
    #     labels = [
    #         label_fn(int(pred) + self.task.label_dictionary.nspecial)
    #         for pred in preds
    #     ]
    #     return [(
    #         token,
    #         label,
    #     ) for token, label in zip(self.tokenize(sentence).split(), labels)]


# In[ ]:


class PororoBertMrc(PororoBiencoderBase):

    def __init__(self, model, tagger, callback, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._callback = callback

    def predict(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Conduct machine reading comprehesion with query and its corresponding context

        Args:
            query: (str) query string used as query
            context: (str) context string used as context
            postprocess: (bool) whether to apply mecab based postprocess

        Returns:
            Tuple[str, Tuple[int, int]]: predicted answer span and its indices

        """
        postprocess = kwargs.get("postprocess", True)

        pair_results = self._model.predict_span(query, context)
        returns = []
        
        for pair_result in pair_results:
            span = self._callback(
            self._tagger,
            pair_result[0],
            ) if postprocess else pair_result[0]
            if len(span) > 1:
                returns.append((span,pair_result[1],pair_result[2]))
        
        returns.sort(key=lambda x:x[2], reverse = True)

        return returns


# In[ ]:


from pororo.models.brainbert import BrainRobertaModel


# In[ ]:


import torch


# In[ ]:


my_mrc_factory = PororoMrcFactory('mrc', 'ko', "brainbert.base.ko.korquad")
my_mrc = my_mrc_factory.load(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# In[ ]:


#log_list = [] # 디버깅용


# 

# # 데이터 탐구

# In[ ]:


train_dataset = load_from_disk('/content/data/data/train_dataset/train')
val_dataset = load_from_disk('/content/data/data/train_dataset/validation')


# ## 다. 는 다음과 같은 방법으로 제끼도록 하자.

# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '다.' in text[:-1]: 
        print(text)


# ## 카카오의 후처리는 " " 이런거 날리는 것 같은데, 좋은지 모르겠다. 외톨이의 '이'도 날린다.

# ## 며, 데, 뿐,  날려도 된다. 라, 고, 이런건 고려해볼 것.

# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '며,' in text: 
        print(text)


# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '데,' in text: 
        print(text)


# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '뿐,' in text: 
        print(text)


# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '라,' in text: 
        print(text)


# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '고,' in text: 
        print(text)


# ## '의'를 날리는건 꽤 모험이었던 것 같다. end 스코어를 확인해서 추가적으로 결정하기? answer 목록에 잘린게 있는지 확인하고 대처하는 건 어떨까?

# In[ ]:


'카에데' in list(map(lambda x: x[0] ,full_mrc))


# In[ ]:


for dict_ in train_dataset:
    text = dict_['answers']['text'][0]
    if '의' in text[-1]:
        wsd_result = wsd(text)[-1]
        if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의'):
            print(text,wsd_result)


# ## 을를이 는 날려도 된다.

# In[ ]:


words = '을를이가는은'
for word in words:
    for dict_ in train_dataset:
        text = dict_['answers']['text'][0]
        if word in text[-1]:
            wsd_result = wsd(text)[-1]
            if (wsd_result.morph == word):
                print(text,wsd_result)


# # 여러 출력 모델 준비

# In[ ]:


bm25_doc=wiki_data['text'].to_numpy()

bm25_score = bm25_ori.get_batch_scores(tokenized_query, range(56737))
bm25_score = np.array(bm25_score)
top_10_score = bm25_score[bm25_score.argsort()[::-1][:10]]
top_10_doc = bm25_doc[bm25_score.argsort()[::-1][:10]]
treshold = top_10_score[0] * 0.95
top_doc = top_10_doc[top_10_score > treshold]
top_doc


# # 리트리버score 기반 커팅(max_*0.95) + 각 문서별 returns concat + MRCscore 상위부터 하나씩 답으로 제출 

# In[ ]:


answer = OrderedDict()
bm25_doc=wiki_data['text'].to_numpy()

for num in tqdm(range(len(test_dataset))):
    id = test_dataset['id'][num]
    query = test_dataset['question'][num]
    tokenized_query = tokenizer(query)

    bm25_score = bm25_ori.get_batch_scores(tokenized_query, range(56737))
    bm25_score = np.array(bm25_score)
    top_10_score = bm25_score[bm25_score.argsort()[::-1][:10]]
    top_10_doc = bm25_doc[bm25_score.argsort()[::-1][:10]]
    treshold = top_10_score[0] * 0.95
    top_doc = top_10_doc[top_10_score > treshold]

    full_mrc = []
    for doc in top_doc:
        full_mrc += my_mrc(query, doc)

    full_mrc.sort(key=lambda x:x[2], reverse = True)
    answers = list(map(lambda x: x[0] ,full_mrc))
    for ans in answers:
        if ('다.' in ans[:-1]) or ('며,' in ans) or ('데,' in ans) or ('뿐,' in ans) or ('라,' in ans) or ('unk' in ans): 
            continue

        # if ans[-1] == '의':
        #     wsd_result = wsd(ans)[-1]
        #     if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의') and ans[:-1] in answers: # '의'가 잘린 답이 후보에 있을 때만
        #         ans = ans[:-1]
        # else:
        words = '의을를이'
        for word in words:
            if ans[-1] == word:
                wsd_result = wsd(ans)[-1]
                if (wsd_result.morph == word):
                    ans = ans[:-1]
                    break

        if (len(ans) <= 1) or (len(ans) >= 30): 
            continue
        print(ans)
        answer[id] = ans
        break

    if not id in answer.keys():
        top_doc = top_10_doc
        full_mrc = []
        for doc in top_doc:
            full_mrc += my_mrc(query, doc)
        full_mrc.sort(key=lambda x:x[2], reverse = True)
        answers = list(map(lambda x: x[0] ,full_mrc))
        for ans in answers:
            if ('다.' in ans[:-1]) or ('며,' in ans) or ('데,' in ans) or ('뿐,' in ans) or ('unk' in ans): 
                continue

            # if ans[-1] == '의': # 은는이가을를
            #     wsd_result = wsd(ans)[-1]
            #     if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의') and ans[:-1] in answers:
            #         ans = ans[:-1]
            # else:
            words = '의을를이'
            for word in words:
                if ans[-1] == word:
                    wsd_result = wsd(ans)[-1]
                    if (wsd_result.morph == word):
                        ans = ans[:-1]
                        break

            if (len(ans) <= 1) or (len(ans) >= 30): 
                continue
            print('top 문서 집합에서 답을 못찾았다::::::::::::::::::::', ans)
            answer[id] = ans
            break
    if not id in answer.keys():
        print('완전히 개처망했다::::::::::::::::::::::::::::::::::::::::::::::::::')

with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/predictions_0506_리트리버스코어기반커팅_returns_concat_MRCsocre순출력.json', 'w') as f:
    json.dump(answer, f)


# 
# ![image.png](data:image/png;base64,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)

# 나이브하게 접근했다가 수정했다.

# # 여러 출력을 내주는 pororo ver2
# 
# start 토큰에 end 보다 가중치를 준다.

# In[ ]:


from typing import Optional, Tuple
from pororo.tasks.utils.base import PororoBiencoderBase, PororoFactoryBase
class PororoMrcFactory(PororoFactoryBase):
    def __init__(self, task: str, lang: str, model: Optional[str]):
        super().__init__(task, lang, model)

    @staticmethod
    def get_available_langs():
        return ["ko"]


    @staticmethod
    def get_available_models():
        return {"ko": ["brainbert.base.ko.korquad"]}


    def load(self, device: str):

        if "brainbert" in self.config.n_model:
            try:
                import mecab
            except ModuleNotFoundError as error:
                raise error.__class__(
                    "Please install python-mecab-ko with: `pip install python-mecab-ko`"
                )
            #from pororo.models.brainbert import BrainRobertaModel
            from pororo.utils import postprocess_span

            model = (My_BrainRobertaModel.load_model(
                f"bert/{self.config.n_model}",
                self.config.lang,
            ).eval().to(device))

            tagger = mecab.MeCab()

            return PororoBertMrc(model, tagger, postprocess_span, self.config)


# In[ ]:


# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import Dict, Tuple, Union

import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer


class My_BrainRobertaModel(RobertaModel):
    """
    Helper class to load pre-trained models easily. And when you call load_hub_model,
    you can use brainbert models as same as RobertaHubInterface of fairseq.
    Methods
    -------
    load_model(log_name: str): Load RobertaModel

    """

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        """
        Load pre-trained model as RobertaHubInterface.
        :param model_name: model name from available_models
        :return: pre-trained model
        """
        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        tok_path = download_or_load(f"tokenizers/bpe32k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )


class BrainRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json",
            merges_filename=f"{tok_path}/merges.txt",
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(s, add_special_tokens=False) +
                             " </s>" if add_special_tokens else "")
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def decode(
        self,
        tokens: torch.LongTensor,
        skip_special_tokens: bool = True,
        remove_bpe: bool = True,
    ) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos(
        ) and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [
                np.array(
                    [c
                     for c in s
                     if c != self.task.source_dictionary.eos()])
                for s in sentences
            ]

        sentences = [
            " ".join([self.task.source_dictionary.symbols[c]
                      for c in s])
            for s in sentences
        ]

        if remove_bpe:
            sentences = [
                s.replace(" ", "").replace("▁", " ").strip() for s in sentences
            ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:
        """
        Predict span from context using a fine-tuned span prediction model.

        :returns answer
            str

        >>> from brain_bert import BrainRobertaModel
        >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korquad')
        >>> model.predict_span(
        ...    'BrainBert는 어떤 언어를 배운 모델인가?',
        ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
        ...    )
        한국어

        """

        max_length = self.task.max_positions()
        tokens = self.encode(
            question,
            context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )[:max_length]
        with torch.no_grad():
            logits = self.predict(
                "span_prediction_head",
                tokens,
                return_logits=True,
            ).squeeze()  # T x 2
            # first predict start position,
            # then predict end position among the remaining logits

            results = []
            # log_list.append(logits) # 디버깅용
            starts = logits[:,0].argsort(descending = True)[:10].tolist()

            for start in starts:
                mask = (torch.arange(
                    logits.size(0), dtype=torch.long, device=self.device) >= start)
                ends = (mask * logits[:, 1]).argsort(descending = True)[:10].tolist()
                for end in ends:
                    answer_tokens = tokens[start:end + 1]
                    
                    answer = ""
                    if len(answer_tokens) >= 1:
                        decoded = self.decode(answer_tokens)
                        if isinstance(decoded, str):
                            answer = decoded

                    score = (logits[:,0][start] + 0.5 * logits[:,0][end]).item()
                    results.append((answer, (start, end + 1), score ))
            
        return results


# In[ ]:


class PororoBertMrc(PororoBiencoderBase):

    def __init__(self, model, tagger, callback, config):
        super().__init__(config)
        self._model = model
        self._tagger = tagger
        self._callback = callback

    def predict(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Conduct machine reading comprehesion with query and its corresponding context

        Args:
            query: (str) query string used as query
            context: (str) context string used as context
            postprocess: (bool) whether to apply mecab based postprocess

        Returns:
            Tuple[str, Tuple[int, int]]: predicted answer span and its indices

        """
        postprocess = kwargs.get("postprocess", True)

        pair_results = self._model.predict_span(query, context)
        returns = []
        
        for pair_result in pair_results:
            span = self._callback(
            self._tagger,
            pair_result[0],
            ) if postprocess else pair_result[0]
            if len(span) > 1:
                returns.append((span,pair_result[1],pair_result[2]))
        
        returns.sort(key=lambda x:x[2], reverse = True)

        return returns


# In[ ]:


from pororo.models.brainbert import BrainRobertaModel


# In[ ]:


import torch


# In[ ]:


my_mrc_factory = PororoMrcFactory('mrc', 'ko', "brainbert.base.ko.korquad")
my_mrc = my_mrc_factory.load(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# # 리트리버score 기반 커팅(max_*0.90) + 각 문서별 returns concat + MRCscore 상위부터 하나씩 답으로 제출 + start에 end보다 더 많은 가중치

# In[ ]:


answer = OrderedDict()
bm25_doc=wiki_data['text'].to_numpy()

for num in tqdm(range(len(test_dataset))):
    id = test_dataset['id'][num]
    query = test_dataset['question'][num]
    tokenized_query = tokenizer(query)

    bm25_score = bm25_ori.get_batch_scores(tokenized_query, range(56737))
    bm25_score = np.array(bm25_score)
    top_10_score = bm25_score[bm25_score.argsort()[::-1][:10]]
    top_10_doc = bm25_doc[bm25_score.argsort()[::-1][:10]]
    treshold = top_10_score[0] * 0.9
    top_doc = top_10_doc[top_10_score > treshold]

    full_mrc = []
    for doc in top_doc:
        full_mrc += my_mrc(query, doc)

    full_mrc.sort(key=lambda x:x[2], reverse = True)
    answers = list(map(lambda x: x[0] ,full_mrc))
    for ans in answers:
        if ('다.' in ans[:-1]) or ('며,' in ans) or ('데,' in ans) or ('뿐,' in ans) or ('라,' in ans) or ('unk' in ans): 
            continue

        # if ans[-1] == '의':
        #     wsd_result = wsd(ans)[-1]
        #     if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의') and ans[:-1] in answers: # '의'가 잘린 답이 후보에 있을 때만
        #         ans = ans[:-1]
        # else:
        words = '의을를이'
        for word in words:
            if ans[-1] == word:
                wsd_result = wsd(ans)[-1]
                if (wsd_result.morph == word):
                    ans = ans[:-1]
                    break

        if (len(ans) <= 1) or (len(ans) >= 30): 
            continue
        print(ans)
        answer[id] = ans
        break

    if not id in answer.keys():
        top_doc = top_10_doc
        full_mrc = []
        for doc in top_doc:
            full_mrc += my_mrc(query, doc)
        full_mrc.sort(key=lambda x:x[2], reverse = True)
        answers = list(map(lambda x: x[0] ,full_mrc))
        for ans in answers:
            if ('다.' in ans[:-1]) or ('며,' in ans) or ('데,' in ans) or ('뿐,' in ans) or ('unk' in ans): 
                continue

            # if ans[-1] == '의': # 은는이가을를
            #     wsd_result = wsd(ans)[-1]
            #     if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의') and ans[:-1] in answers:
            #         ans = ans[:-1]
            # else:
            words = '의을를이'
            for word in words:
                if ans[-1] == word:
                    wsd_result = wsd(ans)[-1]
                    if (wsd_result.morph == word):
                        ans = ans[:-1]
                        break

            if (len(ans) <= 1) or (len(ans) >= 30): 
                continue
            print('top 문서 집합에서 답을 못찾았다::::::::::::::::::::', ans)
            answer[id] = ans
            break
    if not id in answer.keys():
        print('완전히 개처망했다::::::::::::::::::::::::::::::::::::::::::::::::::')

with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/predictions_0506_리트리버스코어기반커팅_returns_concat_MRCsocre순출력_start가중치_09.json', 'w') as f:
    json.dump(answer, f)


# ## 좀 아쉬운 후처리 발견

# In[ ]:


words = '의을를이'
ans = '아편전쟁과 태평천국의 난이'
for word in words:
    if ans[-1] == word:
        wsd_result = wsd(ans)[-1]
        print(wsd_result)
        if (wsd_result.morph == word):
            ans = ans[:-1]
            break


# #여러 출력을 내주는 pororo ver3 (end 토큰은 별 의미가 없다니까??)

# # 리트리버score 기반 커팅(max_*0.9) + 각 문서별 returns concat + MRCscore 상위부터 하나씩 답으로 제출 + end는 걍 무시

# In[ ]:


# Copyright (c) Facebook, Inc., its affiliates and Kakao Brain. All Rights Reserved

from typing import Dict, Tuple, Union

import numpy as np
import torch
from fairseq.models.roberta import RobertaHubInterface, RobertaModel

from pororo.models.brainbert.utils import softmax
from pororo.tasks.utils.download_utils import download_or_load
from pororo.tasks.utils.tokenizer import CustomTokenizer


class My_BrainRobertaModel(RobertaModel):
    """
    Helper class to load pre-trained models easily. And when you call load_hub_model,
    you can use brainbert models as same as RobertaHubInterface of fairseq.
    Methods
    -------
    load_model(log_name: str): Load RobertaModel

    """

    @classmethod
    def load_model(cls, model_name: str, lang: str, **kwargs):
        """
        Load pre-trained model as RobertaHubInterface.
        :param model_name: model name from available_models
        :return: pre-trained model
        """
        from fairseq import hub_utils

        ckpt_dir = download_or_load(model_name, lang)
        tok_path = download_or_load(f"tokenizers/bpe32k.{lang}.zip", lang)

        x = hub_utils.from_pretrained(
            ckpt_dir,
            "model.pt",
            ckpt_dir,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return BrainRobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
            tok_path,
        )


class BrainRobertaHubInterface(RobertaHubInterface):

    def __init__(self, args, task, model, tok_path):
        super().__init__(args, task, model)
        self.bpe = CustomTokenizer.from_file(
            vocab_filename=f"{tok_path}/vocab.json",
            merges_filename=f"{tok_path}/merges.txt",
        )

    def tokenize(self, sentence: str, add_special_tokens: bool = False):
        result = " ".join(self.bpe.encode(sentence).tokens)
        if add_special_tokens:
            result = f"<s> {result} </s>"
        return result

    def encode(
        self,
        sentence: str,
        *addl_sentences,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> torch.LongTensor:
        """
        BPE-encode a sentence (or multiple sentences).
        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.
        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`
        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::
            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        bpe_sentence = self.tokenize(
            sentence,
            add_special_tokens=add_special_tokens,
        )

        for s in addl_sentences:
            bpe_sentence += " </s>" if not no_separator and add_special_tokens else ""
            bpe_sentence += (" " + self.tokenize(s, add_special_tokens=False) +
                             " </s>" if add_special_tokens else "")
        tokens = self.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        return tokens.long()

    def decode(
        self,
        tokens: torch.LongTensor,
        skip_special_tokens: bool = True,
        remove_bpe: bool = True,
    ) -> str:
        assert tokens.dim() == 1
        tokens = tokens.numpy()

        if tokens[0] == self.task.source_dictionary.bos(
        ) and skip_special_tokens:
            tokens = tokens[1:]  # remove <s>

        eos_mask = tokens == self.task.source_dictionary.eos()
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)

        if skip_special_tokens:
            sentences = [
                np.array(
                    [c
                     for c in s
                     if c != self.task.source_dictionary.eos()])
                for s in sentences
            ]

        sentences = [
            " ".join([self.task.source_dictionary.symbols[c]
                      for c in s])
            for s in sentences
        ]

        if remove_bpe:
            sentences = [
                s.replace(" ", "").replace("▁", " ").strip() for s in sentences
            ]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    @torch.no_grad()
    def predict_span(
        self,
        question: str,
        context: str,
        add_special_tokens: bool = True,
        no_separator: bool = False,
    ) -> Tuple:
        """
        Predict span from context using a fine-tuned span prediction model.

        :returns answer
            str

        >>> from brain_bert import BrainRobertaModel
        >>> model = BrainRobertaModel.load_model('brainbert.base.ko.korquad')
        >>> model.predict_span(
        ...    'BrainBert는 어떤 언어를 배운 모델인가?',
        ...    'BrainBert는 한국어 코퍼스에 학습된 언어모델이다.',
        ...    )
        한국어

        """

        max_length = self.task.max_positions()
        tokens = self.encode(
            question,
            context,
            add_special_tokens=add_special_tokens,
            no_separator=no_separator,
        )[:max_length]
        with torch.no_grad():
            logits = self.predict(
                "span_prediction_head",
                tokens,
                return_logits=True,
            ).squeeze()  # T x 2
            # first predict start position,
            # then predict end position among the remaining logits

            results = []
            # log_list.append(logits) # 디버깅용
            starts = logits[:,0].argsort(descending = True)[:10].tolist()

            for start in starts:
                mask = (torch.arange(
                    logits.size(0), dtype=torch.long, device=self.device) >= start)
                ends = (mask * logits[:, 1]).argsort(descending = True)[:10].tolist()
                for end in ends:
                    answer_tokens = tokens[start:end + 1]
                    
                    answer = ""
                    if len(answer_tokens) >= 1:
                        decoded = self.decode(answer_tokens)
                        if isinstance(decoded, str):
                            answer = decoded

                    score = (logits[:,0][start] + 0.1 * logits[:,0][end]).item()
                    results.append((answer, (start, end + 1), score ))
            
        return results


# In[ ]:


from pororo.models.brainbert import BrainRobertaModel


# In[ ]:


import torch


# In[ ]:


my_mrc_factory = PororoMrcFactory('mrc', 'ko', "brainbert.base.ko.korquad")
my_mrc = my_mrc_factory.load(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))


# In[ ]:


answer = OrderedDict()
bm25_doc=wiki_data['text'].to_numpy()

for num in tqdm(range(len(test_dataset))):
    id = test_dataset['id'][num]
    query = test_dataset['question'][num]
    tokenized_query = tokenizer(query)

    bm25_score = bm25_ori.get_batch_scores(tokenized_query, range(56737))
    bm25_score = np.array(bm25_score)
    top_10_score = bm25_score[bm25_score.argsort()[::-1][:10]]
    top_10_doc = bm25_doc[bm25_score.argsort()[::-1][:10]]
    treshold = top_10_score[0] * 0.9
    top_doc = top_10_doc[top_10_score > treshold]

    full_mrc = []
    for doc in top_doc:
        full_mrc += my_mrc(query, doc)

    full_mrc.sort(key=lambda x:x[2], reverse = True)
    answers = list(map(lambda x: x[0] ,full_mrc))
    for ans in answers:
        if ('다.' in ans[:-1]) or ('며,' in ans) or ('데,' in ans) or ('뿐,' in ans) or ('라,' in ans) or ('unk' in ans): 
            continue

        # if ans[-1] == '의':
        #     wsd_result = wsd(ans)[-1]
        #     if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의') and ans[:-1] in answers: # '의'가 잘린 답이 후보에 있을 때만
        #         ans = ans[:-1]
        # else:
        words = '의을를이'
        for word in words:
            if ans[-1] == word:
                wsd_result = wsd(ans)[-1]
                if (wsd_result.morph == word):
                    ans = ans[:-1]
                    break

        if (len(ans) <= 1) or (len(ans) >= 30): 
            continue
        print(ans)
        answer[id] = ans
        break

    if not id in answer.keys():
        top_doc = top_10_doc
        full_mrc = []
        for doc in top_doc:
            full_mrc += my_mrc(query, doc)
        full_mrc.sort(key=lambda x:x[2], reverse = True)
        answers = list(map(lambda x: x[0] ,full_mrc))
        for ans in answers:
            if ('다.' in ans[:-1]) or ('며,' in ans) or ('데,' in ans) or ('뿐,' in ans) or ('unk' in ans): 
                continue

            # if ans[-1] == '의': # 은는이가을를
            #     wsd_result = wsd(ans)[-1]
            #     if (wsd_result.pos == 'JKG') and (wsd_result.morph == '의') and ans[:-1] in answers:
            #         ans = ans[:-1]
            # else:
            words = '의을를이'
            for word in words:
                if ans[-1] == word:
                    wsd_result = wsd(ans)[-1]
                    if (wsd_result.morph == word):
                        ans = ans[:-1]
                        break

            if (len(ans) <= 1) or (len(ans) >= 30): 
                continue
            print('top 문서 집합에서 답을 못찾았다::::::::::::::::::::', ans)
            answer[id] = ans
            break
    if not id in answer.keys():
        print('완전히 개처망했다::::::::::::::::::::::::::::::::::::::::::::::::::')

with open('/content/drive/MyDrive/Colab_data/ai_boostcamp_data/p-stage-3/predictions_0506_리트리버스코어기반커팅_returns_concat_MRCsocre순출력_end거의무시_09.json', 'w') as f:
    json.dump(answer, f)


# In[ ]:




