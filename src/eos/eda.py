"""Загрузка с проверкой, поиск аномалий и стыков областей (раздел 1.1).

``build_dataset`` — полный конвейер: читает все области, проверяет структуру
сетки, помечает аномалии (значения не меняются), считает расхождения на стыках
областей, веса по площади ячеек, общее разбиение ``model.make_split`` и нормировку.
Порядок строк ``data_df`` совпадает с ``data.load_all``, поэтому индексы
``split.npz`` подходят к обеим таблицам.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eos.model import make_split
from eos.normalize import EOSNormalizer

EXPECTED_REGIONS = [f'Obl{k:02d}' for k in range(1, 21)]
_FLOAT = r'([-+0-9.eE]+)'


@dataclass
class RegionMeta:
    """Метаданные области из заголовка файла."""
    region: str
    path: str
    rho_range: tuple
    T_range: tuple
    grid_rho: str
    grid_T: str
    n_declared: int
    n_rho: int = 0
    n_T: int = 0
    warnings: list = field(default_factory=list)

    @property
    def band_rho(self):
        """Номер диапазона плотности 0..3 (по номеру области)."""
        return (int(self.region[3:]) - 1) // 5

    @property
    def band_T(self):
        """Номер диапазона температуры 0..4 (по номеру области)."""
        return (int(self.region[3:]) - 1) % 5


def parse_region_file(path):
    """Читает файл области: (RegionMeta, DataFrame[rho, T, P, e])."""
    header = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.startswith('#'):
                break
            header.append(line.strip())
    h = '\n'.join(header)

    def grab(pattern):
        """Группы регулярного выражения из заголовка; ошибка, если строки нет."""
        m = re.search(pattern, h)
        if m is None:
            raise ValueError(f'{path}: в заголовке нет "{pattern}"')
        return m.groups()

    meta = RegionMeta(
        region=grab(r'Region\s+(Obl\d+)')[0], path=path,
        rho_range=tuple(map(float, grab(r'rho range:\s*\[' + _FLOAT + r',\s*' + _FLOAT))),
        T_range=tuple(map(float, grab(r'T range:\s*\[' + _FLOAT + r',\s*' + _FLOAT))),
        grid_rho=grab(r'Grid type:\s*rho=(\w+),\s*T=(\w+)')[0],
        grid_T=grab(r'Grid type:\s*rho=(\w+),\s*T=(\w+)')[1],
        n_declared=int(grab(r'N points:\s*(\d+)')[0]))

    df = pd.read_csv(path, comment='#', sep=r'\s+', header=None, engine='c')
    if df.shape[1] != 4:
        raise ValueError(f'{path}: ожидалось 4 колонки, получено {df.shape[1]}')
    df.columns = ['rho', 'T', 'P', 'e']
    return meta, df.astype(float)


def _spacing_is(nodes, kind):
    x = np.log10(nodes) if kind == 'log' else nodes
    d = np.diff(x)
    return np.std(d) <= 1e-3 * np.abs(np.mean(d))


def validate_region(meta, df):
    """Структурная проверка области, добавляет индексы узлов i_rho, i_T."""
    name = meta.region
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f'{name}: есть NaN или inf')
    for c in ['rho', 'T', 'P']:
        if (df[c] <= 0).any():
            raise ValueError(f'{name}: {c} <= 0 в {(df[c] <= 0).sum()} точках')
    if len(df) != meta.n_declared:
        meta.warnings.append(f'точек {len(df)}, заявлено {meta.n_declared}')

    rho_u, i_rho = np.unique(df['rho'].to_numpy(), return_inverse=True)
    T_u, i_T = np.unique(df['T'].to_numpy(), return_inverse=True)
    meta.n_rho, meta.n_T = len(rho_u), len(T_u)
    if meta.n_rho * meta.n_T != len(df):
        raise ValueError(f'{name}: сетка не прямоугольная ({meta.n_rho}×{meta.n_T} != {len(df)})')
    if pd.MultiIndex.from_arrays([i_rho, i_T]).has_duplicates:
        raise ValueError(f'{name}: повторяющиеся узлы (rho, T)')

    if not _spacing_is(rho_u, meta.grid_rho):
        meta.warnings.append(f'шаг по rho не соответствует типу сетки {meta.grid_rho}')
    if not _spacing_is(T_u, meta.grid_T):
        meta.warnings.append(f'шаг по T не соответствует типу сетки {meta.grid_T}')
    for var, nodes, (lo, hi) in [('rho', rho_u, meta.rho_range), ('T', T_u, meta.T_range)]:
        if not np.isclose(nodes[0], lo, rtol=1e-6):
            meta.warnings.append(f'{var}_min={nodes[0]:.6g}, заявлено {lo:.6g}')
        if nodes[-1] > hi * (1 + 1e-6):
            meta.warnings.append(f'{var}_max={nodes[-1]:.6g} больше заявленного {hi:.6g}')

    out = df.copy()
    out['i_rho'], out['i_T'] = i_rho, i_T
    return out


def load_regions(data_dir, pattern='Obl*.txt'):
    """Загружает все области; одинаковые файлы одной области (Obl06_1 и т.п.) учитываются один раз."""
    paths = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f'нет файлов {pattern} в {data_dir}')
    regions, hashes, log = {}, {}, []
    for p in paths:
        meta, df = parse_region_file(p)
        h = hashlib.md5(np.ascontiguousarray(df.to_numpy()).tobytes()).hexdigest()
        if meta.region in regions:
            if hashes[meta.region] == h:
                log.append(f'{os.path.basename(p)}: дубликат {meta.region}, пропущен')
                continue
            raise ValueError(f'{meta.region}: {regions[meta.region][0].path} и {p} содержат разные данные')
        regions[meta.region] = (meta, validate_region(meta, df))
        hashes[meta.region] = h
    missing = [r for r in EXPECTED_REGIONS if r not in regions]
    if missing:
        log.append(f'нет областей: {", ".join(missing)}')
    return dict(sorted(regions.items())), log


def _to_grid(df, col, n_rho, n_T):
    g = np.empty((n_T, n_rho))
    g[df['i_T'].to_numpy(), df['i_rho'].to_numpy()] = df[col].to_numpy()
    return g


def _kappa(g, axis):
    g = np.moveaxis(g, axis, 0)
    k = np.zeros_like(g)
    r = g[1:-1] - 0.5 * (g[:-2] + g[2:])
    s = 0.5 * np.abs(g[2:] - g[:-2])
    k[1:-1] = np.abs(r) / (s + 1e-7 * (g.max() - g.min()) + 1e-12)
    return np.moveaxis(k, 0, axis)


def _resid(g, axis):
    g = np.moveaxis(g, axis, 0)
    r = np.zeros_like(g)
    r[1:-1] = np.abs(g[1:-1] - 0.5 * (g[:-2] + g[2:]))
    return np.moveaxis(r, 0, axis)


def detect_anomalies(meta, df, E0=1.0, spike_thr=0.5, spike_min_amp=1e-4,
                     disc_thr=0.2, disc_frac=0.3, disc_min_amp=1e-4):
    """Флаги аномалий (значения не меняются).

    flag_spike — выброс одновременно по rho и по T;
    flag_disc — окрестность разрыва/излома по T (значимого);
    flag_plateau — P и e совпадают с предыдущим узлом по T (холодный предел).
    """
    nr, nT = meta.n_rho, meta.n_T
    qP = np.log10(_to_grid(df, 'P', nr, nT))
    qE = np.arcsinh(_to_grid(df, 'e', nr, nT) / E0)
    T_u = np.sort(df['T'].unique())

    kT = np.maximum(_kappa(qP, 0), _kappa(qE, 0))
    kR = np.maximum(_kappa(qP, 1), _kappa(qE, 1))
    aT = np.maximum(_resid(qP, 0), _resid(qE, 0))
    aR = np.maximum(_resid(qP, 1), _resid(qE, 1))
    spike = (kT > spike_thr) & (kR > spike_thr) & (np.minimum(aT, aR) > spike_min_amp)

    bad_rows = np.where((kT > disc_thr).mean(axis=1) > disc_frac)[0]
    disc = np.zeros((nT, nr), dtype=bool)
    events = []
    if len(bad_rows):
        for gr in np.split(bad_rows, np.where(np.diff(bad_rows) > 1)[0] + 1):
            lo, hi = max(gr[0] - 1, 0), min(gr[-1] + 1, nT - 1)
            dP = np.abs(np.diff(qP[lo:hi + 1], axis=0)).max(axis=0)
            dE = np.abs(np.diff(qE[lo:hi + 1], axis=0)).max(axis=0)
            nb = np.r_[max(lo - 3, 0):lo, hi + 1:min(hi + 4, nT)]
            nb = nb[nb < nT - 1]
            typ = 'излом'
            if len(nb):
                bP = np.median(np.abs(qP[nb + 1] - qP[nb]), axis=0) + 1e-12
                bE = np.median(np.abs(qE[nb + 1] - qE[nb]), axis=0) + 1e-12
                if max(np.median(dP / bP), np.median(dE / bE)) > 3:
                    typ = 'разрыв'
            significant = max(np.median(dP), np.median(dE)) > disc_min_amp
            if significant:
                disc[lo:hi + 1] = True
            events.append({'region': meta.region, 'type': typ,
                           'T_from': float(T_u[lo]), 'T_to': float(T_u[hi]),
                           'cols_frac': float((kT[gr] > disc_thr).mean()),
                           'rel_dP_median': float(np.median(10 ** dP - 1)),
                           'd_asinh_e_median': float(np.median(dE)),
                           'flagged': bool(significant)})

    P_g, e_g = _to_grid(df, 'P', nr, nT), _to_grid(df, 'e', nr, nT)
    plateau = np.zeros((nT, nr), dtype=bool)
    plateau[1:] = (P_g[1:] == P_g[:-1]) & (e_g[1:] == e_g[:-1])

    spike &= ~disc
    out = df.copy()
    it, ir = out['i_T'].to_numpy(), out['i_rho'].to_numpy()
    out['flag_spike'] = spike[it, ir]
    out['flag_disc'] = disc[it, ir]
    out['flag_plateau'] = plateau[it, ir]
    return out, events


def seam_report(regions, E0=1.0):
    """Расхождение P и e на стыках соседних областей (экстраполяция крайнего ряда)."""
    by_band = {(m.band_rho, m.band_T): (m, d) for m, d in regions.values()}
    rows = []
    for (br, bt), (m1, d1) in by_band.items():
        for axis, nxt in [('T', (br, bt + 1)), ('rho', (br + 1, bt))]:
            if nxt not in by_band:
                continue
            m2, d2 = by_band[nxt]
            along, other = ('T', 'rho') if axis == 'T' else ('rho', 'T')
            a_nodes = np.sort(d1[along].unique())
            a2 = np.log10(a_nodes[-2:])
            b = np.log10(d2[along].min())
            first2 = d2[d2[along] == d2[along].min()].sort_values(other)
            res = {}
            for col in ['P', 'e']:
                lines = []
                for val in a_nodes[-2:]:
                    s = d1[d1[along] == val].sort_values(other)
                    y = np.log10(s[col].to_numpy()) if col == 'P' else s[col].to_numpy()
                    lines.append(np.interp(np.log10(first2[other]), np.log10(s[other]), y))
                ext = lines[1] + (lines[1] - lines[0]) * (b - a2[1]) / (a2[1] - a2[0])
                if col == 'P':
                    res[col] = np.abs(10 ** ext / first2['P'].to_numpy() - 1)
                else:
                    res[col] = np.abs(ext - first2['e'].to_numpy()) / (np.abs(first2['e'].to_numpy()) + E0)
            rows.append({'from': m1.region, 'to': m2.region, 'axis': axis, 'boundary': float(10 ** b),
                         'P_rel_median': np.median(res['P']), 'P_rel_max': res['P'].max(),
                         'e_rel_median': np.median(res['e']), 'e_rel_max': res['e'].max()})
    return pd.DataFrame(rows)


def add_weights_and_split(df, val_frac=0.2, seed=0):
    """Веса по площади ячейки в (log rho, log T) и общее разбиение ``make_split``.

    split — 'train' / 'val' / 'test' как в split.npz (train — прореженная сетка,
    на ней же строится бейзлайн); use_train = train без выбросов.
    """
    out = df.copy()
    w = np.empty(len(out))
    for _, idx in out.groupby('region').groups.items():
        sub = out.loc[idx]
        lr = np.log10(np.sort(sub['rho'].unique()))
        lt = np.log10(np.sort(sub['T'].unique()))
        pos = out.index.get_indexer(idx)
        w[pos] = np.gradient(lr)[sub['i_rho']] * np.gradient(lt)[sub['i_T']]
    split = np.empty(len(out), dtype=object)
    for name, rows in make_split(out, val_frac=val_frac, seed=seed).items():
        split[rows] = name
    out['split'] = split
    out['use_train'] = (out['split'] == 'train') & ~out['flag_spike']
    out['w'] = w / w[out['use_train'].to_numpy()].mean()
    return out


def build_dataset(data_dir, E0=1.0, val_frac=0.2, seed=0, verbose=True):
    """Полный конвейер: (data_df, meta_df, report, normalizer)."""
    regions, load_log = load_regions(data_dir)
    frames, events = [], []
    for name, (meta, df) in regions.items():
        df, ev = detect_anomalies(meta, df, E0=E0)
        regions[name] = (meta, df)
        events += ev
        df.insert(0, 'region', name)
        frames.append(df)
    data_df = pd.concat(frames, ignore_index=True)

    seams = seam_report(regions, E0=E0)
    data_df = add_weights_and_split(data_df, val_frac=val_frac, seed=seed)

    data_df['log_rho'] = np.log10(data_df['rho'])
    data_df['log_T'] = np.log10(data_df['T'])
    data_df['log_P'] = np.log10(data_df['P'])
    data_df['asinh_e'] = np.arcsinh(data_df['e'] / E0)

    tr = data_df['use_train']
    normalizer = EOSNormalizer(E0=E0).fit(data_df.loc[tr, 'rho'], data_df.loc[tr, 'T'],
                                          data_df.loc[tr, 'P'], data_df.loc[tr, 'e'])
    data_df[['x_rho', 'x_T']] = normalizer.transform_x(data_df['rho'], data_df['T'])
    data_df[['yP', 'yE']] = normalizer.transform_y(data_df['P'], data_df['e'])
    data_df['region'] = data_df['region'].astype('category')
    data_df[['i_rho', 'i_T']] = data_df[['i_rho', 'i_T']].astype('int32')

    agg = data_df.groupby('region', observed=True).agg(
        n_spike=('flag_spike', 'sum'), n_disc=('flag_disc', 'sum'), n_plateau=('flag_plateau', 'sum'))
    meta_df = pd.DataFrame([{
        'region': m.region, 'n_points': len(d), 'n_points_decl': m.n_declared,
        'n_rho': m.n_rho, 'n_T': m.n_T, 'grid_rho': m.grid_rho, 'grid_T': m.grid_T,
        'rho_min': d['rho'].min(), 'rho_max': d['rho'].max(), 'T_min': d['T'].min(), 'T_max': d['T'].max(),
        'P_min': d['P'].min(), 'P_max': d['P'].max(), 'e_min': d['e'].min(), 'e_max': d['e'].max(),
        'warnings': '; '.join(m.warnings)} for m, d in regions.values()])
    meta_df = meta_df.merge(agg, left_on='region', right_index=True)

    report = {'load_log': load_log, 'events': pd.DataFrame(events), 'seams': seams,
              'split_counts': data_df['split'].value_counts().to_dict()}
    if verbose:
        print(f'Областей: {len(meta_df)}, точек: {len(data_df):,}')
        for line in load_log:
            print('  ', line)
        for _, r in meta_df[meta_df['warnings'] != ''].iterrows():
            print(f'   {r.region}: {r.warnings}')
        print(f"Выбросов: {int(data_df['flag_spike'].sum())}, "
              f"у разрывов/изломов: {int(data_df['flag_disc'].sum())}, "
              f"на плато низких T: {int(data_df['flag_plateau'].sum())}")
        print('Разбиение:', report['split_counts'])
    return data_df, meta_df, report, normalizer
