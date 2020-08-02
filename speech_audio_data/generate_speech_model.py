from scipy.io import wavfile
import pandas as pd
import numpy as np
from joblib import dump, load
from datetime import datetime

def audio_to_dataframe(path):
    from scipy.io import wavfile
    
    sample_rate, data = wavfile.read(path)
    ret_obj = (
        pd.DataFrame(data, columns=['amplitude'])
          .assign(time_id = range(len(data)),
                  file_id = path)
    )
    return ret_obj

import glob

def sample_files(path, frac):
    return pd.Series(glob.glob(path + '/*')).sample(frac=frac)

print('Reading data...')
sample_frac = 0.75
wav_files = pd.concat([sample_files('sounds/dog', sample_frac),
                       sample_files('sounds/cat', sample_frac),
                       sample_files('sounds/bird', sample_frac)])
all_audio = pd.concat([audio_to_dataframe(path) for path in wav_files])
all_labels = pd.Series([wav_file.split('/')[1] for wav_file in wav_files], 
                        index = wav_files)
print(all_audio.head())

regenerate_tsfresh=True
if regenerate_tsfresh:
    print('Generating tsfresh data...')
    from tsfresh import extract_relevant_features, extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    settings = EfficientFCParameters()
#     from tsfresh.feature_extraction import ComprehensiveFCParameters
#     settings = ComprehensiveFCParameters()
    # from tsfresh.feature_extraction import MinimalFCParameters
    # settings = MinimalFCParameters()

    ## TODO: probeer hier ook eens niet relevant_features, maar laat alles uitzoeken door de logreg
#    audio_tsfresh = extract_relevant_features(all_audio, all_labels, 
    audio_tsfresh = extract_features(all_audio, 
                                              column_id='file_id', column_sort='time_id', 
                                              default_fc_parameters=settings)
else:
    print('Reading tsfresh data...')
    all_labels = pd.read_pickle('pkl/speech_tsfresh_labels.pkl')
    audio_tsfresh = pd.read_pickle('pkl/speech_tsfresh.pkl')

print('Running logistic regression CV...')
print('Started CV %s' % datetime.now())
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import l1_min_c

cs = l1_min_c(audio_tsfresh, all_labels, loss='log') * np.logspace(0, 7, 16)
cv_result = LogisticRegressionCV(Cs=cs,
                     penalty='l1', 
                     multi_class='ovr',
                     solver='saga',
                     tol=1e-6,
                     max_iter=int(1e6),
                     n_jobs=-1).fit(audio_tsfresh, all_labels)
print('Done CV %s' % datetime.now())

print('Dumping results...')
all_labels.to_pickle('pkl/speech_tsfresh_labels.pkl')
audio_tsfresh.to_pickle('pkl/speech_tsfresh.pkl')
dump(cv_result, 'pkl/speech_logreg_cv.joblib')