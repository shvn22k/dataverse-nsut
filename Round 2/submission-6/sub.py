import pandas as pd
SUB1_PATH = 'C:\\Projects\\dataverse-nsut\\Round 2\\submission-1\\B7JYI.csv'
SUB5_PATH = 'C:\\Projects\\dataverse-nsut\\Round 2\\submission-5\\B7JYI.csv'
OUT_SAME_OR_0 = 'B7JYI_same_or_0.csv'
OUT_SAME_OR_1 = 'B7JYI_same_or_1.csv'

def main():
    s1 = pd.read_csv(SUB1_PATH)
    s5 = pd.read_csv(SUB5_PATH)
    s1.columns = [c.lower() for c in s1.columns]
    s5.columns = [c.lower() for c in s5.columns]
    pred_col = 'cardio' if 'cardio' in s1.columns else 'target' if 'target' in s1.columns else None
    if pred_col is None:
        raise ValueError('Could not find prediction column in submission-1 file.')
    pred_col_5 = 'cardio' if 'cardio' in s5.columns else 'target' if 'target' in s5.columns else None
    if pred_col_5 is None:
        raise ValueError('Could not find prediction column in submission-5 file.')
    s1['id'] = s1['id'].astype(str)
    s5['id'] = s5['id'].astype(str)
    merged = pd.merge(s1[['id', pred_col]].rename(columns={pred_col: 'pred1'}), s5[['id', pred_col_5]].rename(columns={pred_col_5: 'pred5'}), on='id', how='inner')
    try:
        merged['id_sort'] = merged['id'].astype(int)
    except Exception:
        merged['id_sort'] = merged['id']
    same = merged['pred1'] == merged['pred5']
    out1_pred = merged['pred1'].where(same, 0)
    out1 = pd.DataFrame({'id': merged['id'], 'cardio': out1_pred.astype(int), 'id_sort': merged['id_sort']}).sort_values('id_sort')
    out1 = out1.drop(columns=['id_sort'])
    out1.to_csv(OUT_SAME_OR_0, index=False)
    out2_pred = merged['pred1'].where(same, 1)
    out2 = pd.DataFrame({'id': merged['id'], 'cardio': out2_pred.astype(int), 'id_sort': merged['id_sort']}).sort_values('id_sort')
    out2 = out2.drop(columns=['id_sort'])
    out2.to_csv(OUT_SAME_OR_1, index=False)
    print(f'Wrote {OUT_SAME_OR_0} and {OUT_SAME_OR_1}.')
if __name__ == '__main__':
    main()
