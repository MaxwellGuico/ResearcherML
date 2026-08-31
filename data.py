"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections, datetime, math
import numpy as np

LABEL = 'long_view'
AUXILIARY_LABELS = ('is_click',)
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个特征域。想加特征就往这里加 —— 这是学生最该动的地方之一。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

def load(data_dir):
    """读日志 + 视频侧特征，返回按划分切好的 dict。"""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                             1 if r['is_click'] != '0' else 0))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS))，以及 field_dims。"""
    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        return [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]

    vocabs = [dict() for _ in FIELDS]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


def encode_candidate(splits, feature_variant='baseline'):
    """Encode an approved candidate feature while preserving ``encode`` as baseline.

    Label-derived features are fitted on train only. Their train-row values use
    leave-one-out statistics so a row can never encode its own target.
    """
    if feature_variant == 'baseline':
        return encode(splits)
    if feature_variant not in {'weekday', 'author_affinity', 'user_history'}:
        raise ValueError(f'unsupported feature variant: {feature_variant}')

    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])
    aggregates = collections.defaultdict(lambda: [0, 0])
    if feature_variant in {'author_affinity', 'user_history'}:
        for row in tr:
            key = (row[1], row[3]) if feature_variant == 'author_affinity' else row[1]
            aggregates[key][0] += int(row[6])
            aggregates[key][1] += 1

    def derived(row, *, leave_one_out=False):
        if feature_variant == 'weekday':
            return str(datetime.datetime.strptime(str(row[0]), '%Y%m%d').weekday())
        key = (row[1], row[3]) if feature_variant == 'author_affinity' else row[1]
        positives, count = aggregates.get(key, (0, 0))
        if leave_one_out:
            positives -= int(row[6])
            count -= 1
        if count <= 0:
            return 'cold'
        rate_bucket = min(4, int((positives / count) * 5))
        count_bucket = min(4, int(math.log2(count + 1)))
        return f'{count_bucket}:{rate_bucket}'

    def raw(row, *, leave_one_out=False):
        base = [row[1], row[2], row[3], row[4], str(int(np.searchsorted(edges, row[5])))]
        return base + [derived(row, leave_one_out=leave_one_out)]

    field_count = len(FIELDS) + 1
    vocabs = [dict() for _ in range(field_count)]
    for row in tr:
        for i, value in enumerate(raw(row, leave_one_out=True)):
            if value not in vocabs[i]:
                vocabs[i][value] = len(vocabs[i])
    unknown = [len(vocab) for vocab in vocabs]
    field_dims = [len(vocab) + 1 for vocab in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    encoded = {}
    for name, rows in splits.items():
        features = np.empty((len(rows), field_count), dtype=np.int32)
        labels = np.empty(len(rows), dtype=np.float32)
        users = []
        for n, row in enumerate(rows):
            values = raw(row, leave_one_out=(name == 'train'))
            for i, value in enumerate(values):
                features[n, i] = vocabs[i].get(value, unknown[i]) + offsets[i]
            labels[n] = row[6]
            users.append(row[1])
        encoded[name] = (features, labels, users)
    return encoded, int(sum(field_dims))


def auxiliary_labels(rows, name='is_click'):
    """Return a permitted outcome used only as training-time auxiliary supervision.

    Auxiliary outcomes are carried by the canonical rows from ``load`` so model
    code never reopens raw CSV files. They are never added to inference features.
    """
    if name not in AUXILIARY_LABELS:
        raise ValueError(f'unsupported auxiliary label: {name}')
    index = 7 + AUXILIARY_LABELS.index(name)
    if any(len(row) <= index for row in rows):
        raise ValueError(f'canonical rows do not contain auxiliary label: {name}')
    return np.asarray([1 if row[index] else 0 for row in rows], dtype=np.float32)
