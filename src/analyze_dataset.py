import warnings
warnings.filterwarnings("ignore")
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_data
import argparse
import os
from collections import Counter
import pandas as pd

def analyze_iemocap_dataset(args):
    # Create plots directory if it doesn't exist
    os.makedirs('plots/dataset_analysis', exist_ok=True)
    
    # Load all splits
    train_data = get_data(args, 'iemocap', 'train')
    valid_data = get_data(args, 'iemocap', 'valid')
    test_data = get_data(args, 'iemocap', 'test')
    
    splits = {
        'Train': train_data,
        'Validation': valid_data,
        'Test': test_data
    }
    
    emotions = ["Neutral", "Happy", "Sad", "Angry"]
    
    # Prepare data for analysis
    split_statistics = {}
    all_labels = {emotion: [] for emotion in emotions}
    
    for split_name, data in splits.items():
        # Fix: Access labels correctly from the dataset object
        labels = data.labels.squeeze(-1).numpy() if torch.is_tensor(data.labels) else data.labels
        
        split_statistics[split_name] = {
            'total_samples': len(data),
            'labels': labels,
            'text_features': data.text.shape,
            'audio_features': data.audio.shape,
            'visual_features': data.vision.shape
        }
        
        # Collect labels for each emotion
        for i, emotion in enumerate(emotions):
            emotion_labels = labels[:, i]
            all_labels[emotion].extend(emotion_labels)

    # 1. Plot label distribution for each emotion across splits
    plt.figure(figsize=(15, 8))
    x = np.arange(len(emotions))
    width = 0.25
    
    for i, (split_name, stats) in enumerate(split_statistics.items()):
        counts = []
        for j in range(4):  # 4 emotions
            emotion_labels = stats['labels'][:, j]
            pos_count = np.sum(emotion_labels == 1)
            counts.append(pos_count)
        
        plt.bar(x + i*width, counts, width, label=split_name)
    
    plt.xlabel('Emotions')
    plt.ylabel('Count of Positive Labels')
    plt.title('Distribution of Positive Labels Across Splits')
    plt.xticks(x + width, emotions)
    plt.legend()
    plt.savefig('plots/dataset_analysis/label_distribution.png')
    plt.close()

    # 2. Create a summary DataFrame
    summary_data = {
        'Emotion': [],
        'Split': [],
        'Total Samples': [],
        'Positive Labels': [],
        'Negative Labels': [],
        'Positive Ratio': []
    }

    for split_name, stats in split_statistics.items():
        for i, emotion in enumerate(emotions):
            emotion_labels = stats['labels'][:, i]
            pos_count = np.sum(emotion_labels == 1)
            neg_count = np.sum(emotion_labels == 0)
            total = len(emotion_labels)
            
            summary_data['Emotion'].append(emotion)
            summary_data['Split'].append(split_name)
            summary_data['Total Samples'].append(total)
            summary_data['Positive Labels'].append(pos_count)
            summary_data['Negative Labels'].append(neg_count)
            summary_data['Positive Ratio'].append(pos_count/total)

    df = pd.DataFrame(summary_data)
    
    # 3. Plot heatmap of label distribution
    plt.figure(figsize=(12, 8))
    pivot_table = df.pivot(index='Emotion', columns='Split', values='Positive Ratio')
    sns.heatmap(pivot_table, annot=True, fmt='.2%', cmap='YlOrRd')
    plt.title('Positive Label Ratio Heatmap')
    plt.savefig('plots/dataset_analysis/label_ratio_heatmap.png')
    plt.close()

    # 4. Feature statistics
    feature_stats = {
        'Modality': ['Text', 'Audio', 'Visual'],
        'Feature Dimension': [
            split_statistics['Train']['text_features'][-1],
            split_statistics['Train']['audio_features'][-1],
            split_statistics['Train']['visual_features'][-1]
        ],
        'Sequence Length': [
            split_statistics['Train']['text_features'][1],
            split_statistics['Train']['audio_features'][1],
            split_statistics['Train']['visual_features'][1]
        ]
    }
    
    feature_df = pd.DataFrame(feature_stats)

    # Save all statistics to a report file
    with open('plots/dataset_analysis/dataset_report.txt', 'w') as f:
        f.write("IEMOCAP Dataset Analysis Report\n")
        f.write("==============================\n\n")
        
        f.write("Dataset Split Statistics:\n")
        f.write("-------------------------\n")
        f.write(df.to_string())
        f.write("\n\n")
        
        f.write("Feature Statistics:\n")
        f.write("-----------------\n")
        f.write(feature_df.to_string())
        f.write("\n\n")
        
        # Calculate and write correlations between emotions
        f.write("Emotion Label Correlations:\n")
        f.write("-------------------------\n")
        all_labels_array = np.column_stack([all_labels[emotion] for emotion in emotions])
        corr_matrix = np.corrcoef(all_labels_array.T)
        for i, emotion1 in enumerate(emotions):
            for j, emotion2 in enumerate(emotions):
                if i < j:
                    f.write(f"{emotion1} - {emotion2}: {corr_matrix[i,j]:.3f}\n")

    print(f"Analysis complete! Check the plots and report in plots/dataset_analysis/")
    return df, feature_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IEMOCAP Dataset Analysis')
    parser.add_argument('--aligned', action='store_true',
                        help='use aligned version of dataset')
    parser.add_argument('--data_path', type=str, default='data',
                        help='path for storing the dataset')
    args = parser.parse_args()
    
    df, feature_df = analyze_iemocap_dataset(args)
