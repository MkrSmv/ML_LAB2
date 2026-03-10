import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import sklearn_crfsuite
import pickle
import os
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

TAG_MAP = {
    0: 'O',
    1: 'B-per', 2: 'I-per',
    3: 'B-gpe', 4: 'I-gpe',
    5: 'B-eve', 6: 'I-eve',
    7: 'B-geo', 8: 'I-geo',
    9: 'B-nat', 10: 'I-nat',
    11: 'B-art', 12: 'I-art',
    13: 'B-tim', 14: 'I-tim',
    15: 'B-org', 16: 'I-org',
}
TAG_MAP_INV = {v: k for k, v in TAG_MAP.items()}


def load_data(filepath):
    """Загрузить CSV, заполнить пропуски."""
    df = pd.read_csv(filepath)
    df['Sentence_id'] = df['Sentence_id'].ffill()
    df['Word'] = df['Word'].fillna('').astype(str)
    df['POS'] = df['POS'].fillna('').astype(str)
    return df


def group_sentences(df, has_tag=True):
    """Сгруппировать по предложениям в список кортежей (word, pos[, tag])."""
    sentences = []
    for sid, grp in df.groupby('Sentence_id'):
        if has_tag:
            sent = list(zip(grp['Word'].astype(str),
                            grp['POS'].astype(str),
                            grp['Tag'].astype(int)))
        else:
            sent = list(zip(grp['Word'].astype(str),
                            grp['POS'].astype(str)))
        sentences.append(sent)
    return sentences


def word_features(sent, i):
    """Признаки для i-го токена: слово, POS, регистр, суффиксы, контекст +-2."""
    word = sent[i][0]
    pos = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
        'word.length': len(word),
        'word.has_hyphen': '-' in word,
        'word.has_dot': '.' in word,
        'pos': pos,
    }

    if i > 0:
        w1, p1 = sent[i - 1][0], sent[i - 1][1]
        features.update({
            '-1:word.lower()': w1.lower(),
            '-1:word.istitle()': w1.istitle(),
            '-1:word.isupper()': w1.isupper(),
            '-1:pos': p1,
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        w1, p1 = sent[i + 1][0], sent[i + 1][1]
        features.update({
            '+1:word.lower()': w1.lower(),
            '+1:word.istitle()': w1.istitle(),
            '+1:word.isupper()': w1.isupper(),
            '+1:pos': p1,
        })
    else:
        features['EOS'] = True

    if i > 1:
        w2, p2 = sent[i - 2][0], sent[i - 2][1]
        features.update({
            '-2:word.lower()': w2.lower(),
            '-2:word.istitle()': w2.istitle(),
            '-2:pos': p2,
        })

    if i < len(sent) - 2:
        w2, p2 = sent[i + 2][0], sent[i + 2][1]
        features.update({
            '+2:word.lower()': w2.lower(),
            '+2:word.istitle()': w2.istitle(),
            '+2:pos': p2,
        })

    return features


def sent_to_features(sent):
    return [word_features(sent, i) for i in range(len(sent))]


def sent_to_labels(sent):
    return [TAG_MAP[t[2]] for t in sent]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  NER Pipeline - CRF (sklearn-crfsuite)")
    print("=" * 60)

    print("\n[1/6] Загрузка train.csv...")
    train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
    print(f"  Строк: {len(train_df)}, предложений: {train_df['Sentence_id'].nunique()}")

    tag_counts = train_df['Tag'].value_counts().sort_index()
    print("\n  Распределение тегов:")
    for tag_id, count in tag_counts.items():
        print(f"    {tag_id:>2} ({TAG_MAP[tag_id]:<6}): {count:>8}  "
              f"({count / len(train_df) * 100:.2f}%)")

    print("\n[2/6] Группировка по предложениям...")
    train_sents = group_sentences(train_df, has_tag=True)
    print(f"  Предложений: {len(train_sents)}")

    print("\n[3/6] Извлечение признаков...")
    t0 = time.time()
    X_train = [sent_to_features(s) for s in train_sents]
    y_train = [sent_to_labels(s) for s in train_sents]
    print(f"  Готово за {time.time() - t0:.1f} сек.")

    print("\n[4/6] Обучение CRF...")
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1, c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=False,
    )
    t0 = time.time()
    crf.fit(X_train, y_train)
    print(f"  Обучение завершено за {time.time() - t0:.1f} сек.")

    model_path = os.path.join(OUTPUT_DIR, 'crf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(crf, f)
    print(f"  Модель сохранена: {model_path}")

    print("\n[5/6] Оценка модели...")
    y_pred_train = crf.predict(X_train)

    labels = [v for v in TAG_MAP.values() if v != 'O']
    y_true_flat = [tag for sent in y_train for tag in sent]
    y_pred_flat = [tag for sent in y_pred_train for tag in sent]

    report = classification_report(y_true_flat, y_pred_flat,
                                   labels=labels, digits=4, zero_division=0)
    print(report)

    report_dict = classification_report(y_true_flat, y_pred_flat,
                                        labels=labels, output_dict=True, zero_division=0)
    macro_f1 = report_dict['macro avg']['f1-score']
    print(f"  Macro F1 (без O): {macro_f1:.4f}")
    print(f"  Порог 0.5: {'пройден' if macro_f1 > 0.5 else 'НЕ пройден'}")

    print("\n[6/6] Предсказание на test.csv...")
    test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))
    print(f"  Строк: {len(test_df)}")

    test_sents = group_sentences(test_df, has_tag=False)
    X_test = [sent_to_features(s) for s in test_sents]
    y_pred_test = crf.predict(X_test)

    pred_tags_flat = [TAG_MAP_INV[tag] for sent in y_pred_test for tag in sent]

    submission = pd.DataFrame({
        'ID': test_df.iloc[:, 0].values,
        'Tag': pred_tags_flat,
    })
    submission_path = os.path.join(OUTPUT_DIR, 'my_submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"  Submission: {submission_path} ({len(submission)} строк)")

    print("\n  Готово.")
    return crf, macro_f1


if __name__ == '__main__':
    main()
