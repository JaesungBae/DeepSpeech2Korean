import os 
import scipy.io.wavfile as wav
from calcmfcc import calcfeat_delta_delta
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
from termcolor import cprint
def read_txt(path='./Lexicon_nonkeyword.txt'):
    f = open(path,'r')
    Korean=[]
    Phoneme=[]
    PhonemeList = []
    for i, txt in enumerate(f.readlines()):
        txt = txt.split(' ')
        Phon_tmp = txt[1:-2]
        #print(txt[0],Phon_tmp)
        Korean.append(txt[0]) # Hangul
        Phoneme.append(Phon_tmp) # English
        if ' ' or '\n' in txt[-2:]:
            pass
        else:
            print(txt[-2:])
            raise ValueError('there is some word in the last 2 elemets')
        for j in range(len(Phon_tmp)):
            if Phon_tmp[j] in PhonemeList:
                pass
            else:
                #print(Phon_tmp[j])
                #print(Phon_tmp)
                PhonemeList.append(Phon_tmp[j])
                #print('PhonemeList appended')
    PhonemeList.sort()
    print('Phoneme List:\n',PhonemeList)
    assert len(Korean) == len(Phoneme)
    f.close()
    # Dont know why but there is some empty space in front of the first elemet of Korean list.
    Korean[0] = Korean[0][1]
    return Korean, Phoneme, PhonemeList

def choose_hangul(txt):
    # There is a chase when there is number or english they give us two choice.
    # We decide to choose the hangul one.
    if '/' in txt:# A[B]/[C]D[E]/[f]G
        new_txt = []
        for aaaa in txt.split('['):# A, B]/, C]D, E]/, F]G
            if '/' in aaaa:
                pass
            else: # A, C]D, F]G
                if ']' in aaaa:
                    for txt_a in aaaa.split(']'):# C D
                        new_txt.append(txt_a)
                else:
                    new_txt.append(aaaa) #A
        new_txt = ''.join(new_txt)
    else:
        new_txt = txt
    return new_txt

def feature_generation(audio_path, save_path, 
    win_length=0.02, win_step=0.01, mode='fbank', feature_len=40, noise_name='clean', noiseSNR=0.5):
    '''
    <input>
    audio_path = '/sda3/DATA/jsbae/Google_Speech_Command'
    save_path = '/home/jsbae/STT2/KWS/feature_saved'
    win_length: default=0.02, "specify the window length of feature"
    win_step: default=0.01, "specify the window step length of feature"
    mode: choices=['mfcc', 'fbank']
    feature_len: default=40,'Features length'
    <output>
    No output. Save featuere and label(int) to npy filetpye.
    '''
    # Keyword/Test/TV 냉장고/number/audio
    # Nonkeyword/Test/TV 냉장고/number/audio and text
    keyword = ['Keyword','Nonkeyword']
    is_training = ['Test','Train']
    noise = ['TV','냉장고']
    Path = []
    for j, is_training_ in enumerate(is_training):
        for k, noise_ in enumerate(noise):
            Path.append(os.path.join(is_training_,noise_))
    for path_ in Path:
        for keyword_ in keyword:
            if not os.path.exists(os.path.join(save_path,mode,keyword_,path_)):
                os.makedirs(os.path.join(save_path,mode,keyword_,path_))
                print('path created')
            if not os.path.exists(os.path.join(save_path,'label',keyword_,path_)):
                os.makedirs(os.path.join(save_path,'label',keyword_,path_))
                print('path created')
    # keyword
    count = 0
    for path_ in Path:
        AudioPath = os.path.join(audio_path,keyword[0],path_)
        SavePath = os.path.join(save_path,keyword[0],path_)
        dirs = [f for f in os.listdir(AudioPath) if os.path.isdir(os.path.join(AudioPath,f))]
        for dirname in dirs:
            full_dirname = os.path.join(AudioPath,dirname)
            teCount, vaCount, trCount = 0,0,0
            for filename in os.listdir(full_dirname):
                full_filename = os.path.join(full_dirname,filename)
                filenameNoSuffix =  os.path.splitext(full_filename)[0]
                ext = os.path.splitext(full_filename)[-1]
                if ext == '.wav':
                    rate = None
                    sig = None
                    (rate,sig) = wav.read(full_filename)
                    feat = calcfeat_delta_delta(sig,rate,win_length=win_length,win_step=win_step,mode=mode,feature_len=feature_len)
                    feat = preprocessing.scale(feat)
                    feat = np.transpose(feat)

                    A = filenameNoSuffix.split('_')[2][-1]
                    if A == '1':
                        #txt = '알파봇'
                        Keyword_phoneme = ['a','xl','p','a','b','o','xd']
                    elif A == '5':
                        #txt = '온누리'
                        Keyword_phoneme = ['o', 'xn', 'n', 'u', 'r', 'i']
                    elif A == '9':
                        #txt = '미리내'
                        Keyword_phoneme = ['m', 'i', 'r', 'i', 'n', 'E']
                    else:
                        raise ValueError
                    phoneme_label = []
                    for phoneme_ in Keyword_phoneme:
                            phoneme_label.append(PhonemeList.index(phoneme_))
                    #print(phoneme_label)
                    # SAVE
                    featureFilename = os.path.join(save_path,mode,keyword[0],path_,filenameNoSuffix.split('/')[-1])+'.npy'
                    labelFilename = os.path.join(save_path,'label',keyword[0],path_,filenameNoSuffix.split('/')[-1])+'.npy'
                    #if os.path.exists(featureFilename) or os.path.exists(labelFilename):
                    #        raise ValueError('Already Exsits file name')
                    #print(featureFilename)
                    #print(labelFilename)
                    #print(feat)
                    #print(phoneme_label)
                    #np.save(featureFilename, feat)
                    np.save(labelFilename, phoneme_label)
                    count += 1
                else:
                    raise ValueError
    print('Keyword wav number: {}'.format(count))
    # Nonekeyword
    connt = 0
    for path_ in Path:
        AudioPath = os.path.join(audio_path,keyword[1],path_)
        SavePath = os.path.join(save_path,keyword[1],path_)
        dirs = [f for f in os.listdir(AudioPath) if os.path.isdir(os.path.join(AudioPath,f))]
        for dirname in dirs:
            full_dirname = os.path.join(AudioPath,dirname)
            teCount, vaCount, trCount = 0,0,0
            for filename in os.listdir(full_dirname):
                full_filename = os.path.join(full_dirname,filename)
                filenameNoSuffix =  os.path.splitext(full_filename)[0]
                ext = os.path.splitext(full_filename)[-1]
                DontSave = 0
                if ext == '.wav':
                    rate = None
                    sig = None
                    (rate,sig) = wav.read(full_filename)
                    feat = calcfeat_delta_delta(sig,rate,win_length=win_length,win_step=win_step,mode=mode,feature_len=feature_len)
                    feat = preprocessing.scale(feat)
                    feat = np.transpose(feat)

                    A = filenameNoSuffix.split('_')[0]
                    txt_filename = A + '.txt'
                    f = open(txt_filename,'r',encoding='euc-kr')#'utf-8') 
                    # Encoded in to 'euc-kr'
                    txt = f.readline()
                    phoneme_label = []
                    # Chosee Hangul in chase if there is English or number
                    new_txt = choose_hangul(txt)
                    for i in new_txt.split(' '):
                        if i in Korean:
                            idx = Korean.index(i)
                            for phoneme_ in Phoneme[idx]:
                                phoneme_label.append(PhonemeList.index(phoneme_))
                        elif i == '':
                            pass
                        elif i == '\n':
                            pass
                        elif i[-1] == '\n':
                            idx = Korean.index(i[:-1])
                            for phoneme_ in Phoneme[idx]:
                                phoneme_label.append(PhonemeList.index(phoneme_))
                        else:
                            # Case that there is word that is not in the lexicon dict.
                            # -> Dont save this audio
                            cprint(i,'red')
                            DontSave = 1
                    phoneme_label = np.array(phoneme_label)
                    #print(phoneme_label)
                    # SAVE
                    if DontSave:
                        cprint('Dont save this audio '+filenameNoSuffix.split('_')[0],'red')
                    else:
                        featureFilename = os.path.join(save_path,mode,keyword[1],path_,filenameNoSuffix.split('/')[-1])+'.npy'
                        labelFilename = os.path.join(save_path,'label',keyword[1],path_,filenameNoSuffix.split('/')[-1])+'.npy'
                        #if os.path.exists(featureFilename) or os.path.exists(labelFilename):
                        #    raise ValueError('Already Exsits file name')
                        #print(featureFilename)
                        #print(labelFilename)
                        #print(feat)
                        #print(phoneme_label)
                        #np.save(featureFilename, feat)
                        np.save(labelFilename, phoneme_label)
                        count += 1
                else:
                    pass
    print('Nonkeyword wav number: {}'.format(count))


Korean,Phoneme,PhonemeList = read_txt()
PhonemeList = ['_'] + PhonemeList
print(PhonemeList)
feature_generation(audio_path='./data', 
    save_path='./feature_saved',
    win_length=0.02, win_step=0.01, 
    mode='fbank', feature_len=40)

