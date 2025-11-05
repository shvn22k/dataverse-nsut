import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
print('=' * 80)
print('SUBMISSION ENSEMBLE - MAJORITY VOTING')
print('=' * 80)
submissions_to_combine = [('C:\\Projects\\dataverse-nsut\\submission-3\\B7JYI-II.csv', 'Original 99.57%'), ('C:\\Projects\\dataverse-nsut\\submission-4\\B7JYI-2.csv', 'Submission-5'), ('C:\\Projects\\dataverse-nsut\\submission-5\\B7JYI.csv', 'Submission-6'), ('C:\\Projects\\dataverse-nsut\\submission-6\\B7JYI.csv', 'Submission-7'), ('C:\\Projects\\dataverse-nsut\\submission-7\\B7JYI.csv', 'Submission-8'), ('C:\\Projects\\dataverse-nsut\\submission-8\\B7JYI.csv', 'Submission-9'), ('C:\\Projects\\dataverse-nsut\\submission-9\\B7JYI.csv', 'Submission-10'), ('C:\\Projects\\dataverse-nsut\\submission-10\\B7JYI.csv', 'Submission-11'), ('C:\\Projects\\dataverse-nsut\\submission-11\\B7JYI-t1.csv', 'Submission-12'), ('C:\\Projects\\dataverse-nsut\\submission-12\\B7JYI.csv', 'Submission-13'), ('C:\\Projects\\dataverse-nsut\\submission-13\\B7JYI-2.csv', 'Submission-14'), ('C:\\Projects\\dataverse-nsut\\submission-14\\B7JYI.csv', 'Submission-15')]
print(f'\n[STEP 1] Loading {len(submissions_to_combine)} submissions...')
print()
dfs = []
for submission_info in submissions_to_combine:
    if len(submission_info) == 3:
        path, desc, score = submission_info
    else:
        path, desc = submission_info
        score = 'Unknown'
    try:
        df = pd.read_csv(path)
        if score == 'Unknown':
            print(f'{desc:<30} Score: Unknown  ({len(df):,} samples)')
        else:
            print(f'{desc:<30} {score:.2f}%  ({len(df):,} samples)')
        dfs.append((df, desc, score))
    except FileNotFoundError:
        print(f'✗ {desc:<30} FILE NOT FOUND: {path}')
        print(f'  Skipping this submission...')
if len(dfs) == 0:
    print('\n No submissions loaded! Please check file paths.')
    exit(1)
if len(dfs) == 1:
    print('\n  Only 1 submission loaded. Ensemble needs at least 2!')
    print('   Add more submission paths to the list above.')
    exit(1)
print(f'\nLoaded {len(dfs)} submissions')
print('\n[STEP 2] Verifying submissions...')
base_ids = dfs[0][0]['sha256'].values
for i, (df, desc, score) in enumerate(dfs[1:], 1):
    if not np.array_equal(df['sha256'].values, base_ids):
        print(f'ERROR: {desc} has different IDs than base submission!')
        print('   All submissions must have same samples in same order.')
        exit(1)
print(f'All submissions have same {len(base_ids):,} IDs in same order')
print('\n[STEP 3] Performing majority voting...')
all_predictions = np.array([df['Label'].values for df, _, _ in dfs])
print(f'Shape: {all_predictions.shape} ({all_predictions.shape[0]} models × {all_predictions.shape[1]} samples)')
final_predictions = []
disagreements = 0
unanimous = 0
for i in range(all_predictions.shape[1]):
    votes = all_predictions[:, i]
    vote_counts = Counter(votes)
    majority_vote = vote_counts.most_common(1)[0][0]
    final_predictions.append(majority_vote)
    if len(vote_counts) > 1:
        disagreements += 1
    if len(vote_counts) == 1:
        unanimous += 1
final_predictions = np.array(final_predictions)
print(f'\nVoting statistics:')
print(f'  Unanimous: {unanimous:,} samples ({unanimous / len(final_predictions) * 100:.2f}%)')
print(f'  Disagreements: {disagreements:,} samples ({disagreements / len(final_predictions) * 100:.2f}%)')
malware_count = np.sum(final_predictions == 1)
benign_count = np.sum(final_predictions == 0)
print(f'\nFinal prediction distribution:')
print(f'  Malware (1): {malware_count:,} ({malware_count / len(final_predictions) * 100:.2f}%)')
print(f'  Benign (0): {benign_count:,} ({benign_count / len(final_predictions) * 100:.2f}%)')
print(f'\nComparison with individual submissions:')
for df, desc, score in dfs:
    agreement = np.sum(df['Label'].values == final_predictions)
    print(f'  {desc:<30} {agreement:,}/{len(final_predictions):,} agree ({agreement / len(final_predictions) * 100:.2f}%)')
print('\n[STEP 4] Analyzing disagreements...')
if disagreements > 0:
    disagreement_indices = []
    for i in range(all_predictions.shape[1]):
        votes = all_predictions[:, i]
        if len(set(votes)) > 1:
            disagreement_indices.append(i)
    print(f'\nDisagreement patterns:')
    patterns = []
    for idx in disagreement_indices:
        votes = tuple(all_predictions[:, idx])
        patterns.append(votes)
    pattern_counts = Counter(patterns)
    print(f'  Unique voting patterns: {len(pattern_counts)}')
    if len(dfs) <= 5:
        print(f'  Top patterns:')
        for pattern, count in pattern_counts.most_common(5):
            print(f'    {pattern}: {count} samples')
else:
    print(' All models agree on all samples (perfect agreement!)')
print('\n[STEP 5] Creating final submission...')
final_submission = pd.DataFrame({'sha256': base_ids, 'Label': final_predictions.astype(int)})
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'submission_ensemble_{len(dfs)}models_{timestamp}.csv'
final_submission.to_csv(filename, index=False)
print('\n' + '=' * 80)
print('ENSEMBLE COMPLETE!')
print('=' * 80)
print(f'\nFile: {filename}')
print(f'Models combined: {len(dfs)}')
print(f'Total samples: {len(final_submission):,}')
print(f'\nModels used:')
for df, desc, score in dfs:
    if score == 'Unknown':
        print(f'  - {desc} (Score: Unknown)')
    else:
        print(f'  - {desc} ({score:.2f}%)')
print(f'\nExpected improvement:')
if dfs[0][2] != 'Unknown':
    print(f'  - Base model: {dfs[0][2]:.2f}%')
    print(f'  - Target: Beat {dfs[0][2]:.2f}%')
else:
    print(f'  - Base model: Unknown score')
    print(f'  - Target: Improve over individual models')
print(f'  - Ensemble typically: +0.05% to +0.20%')
print(f'\nKey insight:')
print(f'  {disagreements:,} samples had disagreements')
print(f'  Majority voting corrected potential errors on these')
print('=' * 80)
