import pandas as pd
print('Comparing OLD data vs NEW data\n')
print('=' * 70)
print('\n1. COMPARING test.csv:')
print('-' * 70)
old_test = pd.read_csv('data/test.csv')
new_test = pd.read_csv('new_data/test.csv')
print(f'Old test shape: {old_test.shape}')
print(f'New test shape: {new_test.shape}')
old_test_ids = set(old_test['sha256'])
new_test_ids = set(new_test['sha256'])
missing_in_new = old_test_ids - new_test_ids
added_in_new = new_test_ids - old_test_ids
print(f'\nUnique IDs in old: {len(old_test_ids)}')
print(f'Unique IDs in new: {len(new_test_ids)}')
if missing_in_new:
    print(f'\n✗ IDs removed in new test: {len(missing_in_new)}')
    print(f'  Examples: {list(missing_in_new)[:5]}')
else:
    print('\n No IDs removed')
if added_in_new:
    print(f'\n✗ IDs added in new test: {len(added_in_new)}')
    print(f'  Examples: {list(added_in_new)[:5]}')
else:
    print('\n No new IDs added')
print('\n' + '=' * 70)
print('\n2. COMPARING sample_submission.csv:')
print('-' * 70)
old_sample = pd.read_csv('data/sample_submission.csv')
new_sample = pd.read_csv('new_data/sample_submission.csv')
print(f'Old sample shape: {old_sample.shape}')
print(f'New sample shape: {new_sample.shape}')
old_sample_ids = set(old_sample['sha256'])
new_sample_ids = set(new_sample['sha256'])
missing_in_new_sample = old_sample_ids - new_sample_ids
added_in_new_sample = new_sample_ids - old_sample_ids
print(f'\nUnique IDs in old: {len(old_sample_ids)}')
print(f'Unique IDs in new: {len(new_sample_ids)}')
if missing_in_new_sample:
    print(f'\n✗ IDs removed in new sample: {len(missing_in_new_sample)}')
    print(f'  Examples: {list(missing_in_new_sample)[:5]}')
else:
    print('\n No IDs removed')
if added_in_new_sample:
    print(f'\n✗ IDs added in new sample: {len(added_in_new_sample)}')
    print(f'  Examples: {list(added_in_new_sample)[:5]}')
else:
    print('\n No new IDs added')
print('\n' + '=' * 70)
print('\n3. COMPARING main.csv:')
print('-' * 70)
old_main = pd.read_csv('data/main.csv')
new_main = pd.read_csv('new_data/main.csv')
print(f'Old main shape: {old_main.shape}')
print(f'New main shape: {new_main.shape}')
old_main_ids = set(old_main['sha256'])
new_main_ids = set(new_main['sha256'])
print(f'\nUnique IDs in old: {len(old_main_ids)}')
print(f'Unique IDs in new: {len(new_main_ids)}')
missing_in_new_main = old_main_ids - new_main_ids
added_in_new_main = new_main_ids - old_main_ids
if missing_in_new_main:
    print(f'\n✗ IDs removed in new main: {len(missing_in_new_main)}')
else:
    print('\n No IDs removed')
if added_in_new_main:
    print(f'\n✗ IDs added in new main: {len(added_in_new_main)}')
else:
    print('\n No new IDs added')
print('\n' + '=' * 70)
print('\n4. CHECKING YOUR SUBMISSION vs NEW DATA:')
print('-' * 70)
submission = pd.read_csv('submission-1/B7JYI.csv')
submission_ids = set(submission['sha256'])
print(f'Your submission IDs: {len(submission_ids)}')
print(f'New sample IDs: {len(new_sample_ids)}')
missing_in_submission = new_sample_ids - submission_ids
extra_in_submission = submission_ids - new_sample_ids
if missing_in_submission:
    print(f'\n✗ PROBLEM! Missing {len(missing_in_submission)} IDs from new sample_submission')
    print(f'  First 10 missing IDs:')
    for mid in list(missing_in_submission)[:10]:
        print(f'    {mid}')
        if mid.startswith('EFF0ADF'):
            print(f'      👆 THIS MATCHES THE ERROR!')
else:
    print('\n All new sample IDs are in your submission')
if extra_in_submission:
    print(f'\n✗ Your submission has {len(extra_in_submission)} extra IDs not in new sample')
    print(f'  First 5 extra IDs:')
    for eid in list(extra_in_submission)[:5]:
        print(f'    {eid}')
else:
    print('\n No extra IDs in submission')
print('\n' + '=' * 70)
print('SUMMARY:')
print('=' * 70)
if missing_in_submission or extra_in_submission:
    print('\n  THE DATA FILES CHANGED!')
    print(f'   Your submission was based on OLD data.')
    print(f'   You need to re-generate predictions using NEW data.')
    print(f'\n   Missing IDs: {(len(missing_in_submission) if missing_in_submission else 0)}')
    print(f'   Extra IDs: {(len(extra_in_submission) if extra_in_submission else 0)}')
else:
    print('\n Data files are the same, no changes detected')
