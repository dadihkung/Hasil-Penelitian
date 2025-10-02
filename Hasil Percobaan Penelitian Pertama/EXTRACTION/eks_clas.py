import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, detrend
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from scipy.stats import entropy

# ===========================
# CONFIGURATION
# ===========================
raw_data_dirs = [
    r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_didan_simdrone",
    r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_fahri_simdrone",
    r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_subagja_simdrone",
    r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_sultan_simdrone",
    r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_syarif_simdrone",
    r"C:\laragon\www\skripsi\EEG MENTAH\OpenBCISession_usamah_simdrone"
]

channels = ["EXG0", "EXG1"]
sfreq = 250  # Hz

# Command timing information
command_times_0 = {
    'depan':     [(0, 7), (72, 79), (144, 151), (216, 223), (288, 295)],
    'belakang': [(12, 19), (84, 91), (156, 163), (228, 235), (300, 307)],
    'kanan':    [(24, 31), (96, 103), (168, 175), (240, 247), (312, 319)],
    'kiri':     [(36, 43), (108, 115), (180, 187), (252, 259), (324, 331)],
    'terbang':  [(48, 55), (120, 127), (192, 199), (264, 271), (336, 343)],
    'landing':  [(60, 67), (132, 139), (204, 211), (276, 283), (348, 355)],
}

command_times_5 = {
    'depan':     [(5, 12), (77, 84), (149, 156), (221, 228), (293, 300)],
    'belakang': [(17, 24), (89, 96), (161, 168), (233, 240), (305, 312)],
    'kanan':    [(29, 36), (101, 108), (173, 180), (245, 252), (317, 324)],
    'kiri':     [(41, 48), (113, 120), (185, 192), (257, 264), (329, 336)],
    'terbang':  [(53, 60), (125, 132), (197, 204), (269, 276), (341, 348)],
    'landing':  [(65, 72), (137, 144), (209, 216), (281, 288), (353, 360)],
}

# Feature extraction parameters
window_sec = 3.0  # Smaller window often works better for Random Forest
overlap = 0.5
n_estimators = 200  # Number of trees in Random Forest
random_state = 42

# ===========================
# SIGNAL PROCESSING FUNCTIONS
# ===========================
def clean_column(col):
    col = col.astype(str).str.replace('.', '', regex=False)
    col = col.str.replace(',', '.', regex=False)
    return pd.to_numeric(col, errors='coerce')

def design_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(x, low, high, fs):
    b, a = design_bandpass(low, high, fs)
    return filtfilt(b, a, x, axis=0)

def apply_notch(x, f0, fs, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, x, axis=0)

def enhanced_preprocess(sig, fs):
    # 1. Remove mean and linear trends
    sig = detrend(sig, axis=0, type='linear')
    
    # 2. Apply notch filter at 50Hz and harmonics
    sig = apply_notch(sig, 50, fs)
    sig = apply_notch(sig, 100, fs)  # 2nd harmonic
    
    # 3. Bandpass filter
    sig = apply_bandpass(sig, 5, 40, fs)
    
    # 4. Common average reference
    sig = sig - np.mean(sig, axis=1, keepdims=True)
    
    # 5. Clip extreme values
    sig = np.clip(sig, -100, 100)
    
    # 6. Standardize
    sig = (sig - np.mean(sig)) / np.std(sig)
    
    return sig

# ===========================
# FEATURE EXTRACTION
# ===========================
def extract_psd_features(sig, fs):
    """Extract power spectral density features"""
    f, Pxx = signal.welch(sig, fs=fs, nperseg=fs*2, axis=0)
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    features = []
    for band, (low, high) in bands.items():
        mask = (f >= low) & (f <= high)
        band_power = np.mean(Pxx[mask], axis=0)
        features.extend(band_power)
    
    return np.array(features)

def extract_time_features(sig):
    """Extract time-domain features"""
    features = []
    for ch in range(sig.shape[1]):
        ch_data = sig[:, ch]
        features.extend([
            np.mean(ch_data),
            np.std(ch_data),
            np.min(ch_data),
            np.max(ch_data),
            np.median(ch_data),
            np.mean(np.abs(ch_data - np.mean(ch_data))),  # MAD
            np.percentile(ch_data, 25),
            np.percentile(ch_data, 75),
            entropy(ch_data)  # Shannon entropy
        ])
    return np.array(features)

def extract_features(sig, fs):
    """Combine PSD and time-domain features"""
    psd_features = extract_psd_features(sig, fs)
    time_features = extract_time_features(sig)
    return np.concatenate([psd_features, time_features])

# ===========================
# DATA LOADING & PROCESSING
# ===========================
def load_raw_eeg_file(file_path):
    """Load and preprocess a single raw EEG file"""
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    df = df.iloc[:, 0:3]  # Keep only relevant columns
    df.columns = ['Sample', 'EXG0', 'EXG1']
    
    # Clean and convert data
    for col in ['EXG0', 'EXG1']:
        df[col] = clean_column(df[col])
    
    # Remove bad samples
    df = df.dropna()
    return df

def extract_command_segments(subject_name, raw_df):
    """Extract command segments from raw EEG data"""
    # Determine which command times to use based on subject
    if subject_name.lower() in ['subagja', 'didan']:
        current_command_times = command_times_5
    else:
        current_command_times = command_times_0

    # Preprocess the entire recording
    eeg_data = raw_df[['EXG0', 'EXG1']].values
    eeg_data = enhanced_preprocess(eeg_data, sfreq)

    command_segments = []
    
    for command, time_ranges in current_command_times.items():
        for loop_num, (tmin, tmax) in enumerate(time_ranges, 1):
            start_idx = int(tmin * sfreq)
            end_idx = int(tmax * sfreq)
            
            if start_idx >= len(eeg_data):
                continue
            if end_idx > len(eeg_data):
                end_idx = len(eeg_data)
                
            segment = eeg_data[start_idx:end_idx, :]
            
            seg_df = pd.DataFrame(segment, columns=['EXG0', 'EXG1'])
            seg_df['subject'] = subject_name
            seg_df['loop'] = loop_num
            seg_df['time'] = np.arange(0, len(segment)) / sfreq + tmin
            seg_df['command'] = command
            
            command_segments.append(seg_df)
    
    return command_segments

def create_windows_and_features(command_df):
    """Create analysis windows and extract features"""
    win_size = int(window_sec * sfreq)
    step = int(win_size * (1 - overlap))
    
    features = []
    labels = []
    subjects = []
    
    grouped = command_df.groupby(['subject', 'command', 'loop'])
    
    for (subject, command, loop), group in grouped:
        eeg_data = group[['EXG0', 'EXG1']].values
        
        for start in range(0, len(eeg_data) - win_size + 1, step):
            window = eeg_data[start:start + win_size, :]
            
            # Extract features for this window
            window_features = extract_features(window, sfreq)
            features.append(window_features)
            labels.append(command)
            subjects.append(subject)
    
    return np.array(features), np.array(labels), np.array(subjects)

# ===========================
# MODEL TRAINING & EVALUATION
# ===========================
def train_and_evaluate(X, y, groups):
    """Enhanced training with feature selection and class balancing"""
    logo = LeaveOneGroupOut()
    
    # Update model pipeline
    model = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),  # Keep 95% variance
        SelectKBest(f_classif, k=10),  # Select top 10 features
        SMOTE(random_state=random_state),  # Handle class imbalance
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=random_state
        )
    )
    
    accuracies = []
    y_true_all = []
    y_pred_all = []
    feature_importances = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store feature importances
        if hasattr(model.named_steps['randomforestclassifier'], 'feature_importances_'):
            feature_importances.append(
                model.named_steps['randomforestclassifier'].feature_importances_)
        
        acc = accuracy_score(y_test, y_pred)
        subject = groups[test_idx[0]]
        print(f"Subject {subject}: Accuracy = {acc:.2f}")
        
        accuracies.append(acc)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
    
    # Feature importance analysis
    if feature_importances:
        avg_importances = np.mean(feature_importances, axis=0)
        print("\nTop 10 Most Important Features:")
        # You should replace this with your actual feature names
        for i in np.argsort(avg_importances)[-10:][::-1]:
            print(f"Feature {i}: {avg_importances[i]:.4f}")
    
    print("\nEnhanced Performance:")
    print(f"Mean Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all))
    
    plot_confusion_matrix(y_true_all, y_pred_all)
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(y_true)
    
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ===========================
# MAIN EXECUTION
# ===========================
if __name__ == "__main__":
    print("Loading and processing EEG data...")
    
    # Load all data
    all_segments = []
    for data_dir in raw_data_dirs:
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv') and 'BrainFlow-RAW' in fname:
                file_path = os.path.join(data_dir, fname)
                subject_name = fname.split('_')[-2]
                
                print(f"Processing {subject_name}...")
                try:
                    raw_df = load_raw_eeg_file(file_path)
                    segments = extract_command_segments(subject_name, raw_df)
                    all_segments.extend(segments)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    if not all_segments:
        raise ValueError("No EEG data was loaded - check your file paths")
    
    command_df = pd.concat(all_segments, ignore_index=True)
    
    # Extract features
    print("\nExtracting features...")
    X, y, subjects = create_windows_and_features(command_df)
    
    # Train and evaluate
    print("\nTraining and evaluating Random Forest model...")
    model = train_and_evaluate(X, y, subjects)
    
    # Optional: Save model for later use
    # import joblib
    # joblib.dump(model, 'eeg_classifier_rf.pkl')