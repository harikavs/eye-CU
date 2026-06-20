#!/usr/bin/env python3
"""
analyse_session.py  —  Post-session analysis for the Cognitive Strain Detector.

Picks up the latest CSV in ../logs/ (or a path from argv[1]), produces a
7-panel figure saved as a PNG next to the CSV, and prints a text summary.

Usage
-----
    python analyse_session.py                   # latest session in ../logs/
    python analyse_session.py path/to/file.csv  # explicit file
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

# ── Config ────────────────────────────────────────────────────────────────────
LOGS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'logs')
SMOOTH_WIN   = 90   # frames (~3 s at 30 fps)

STRAIN_COLS  = {
    'LOW STRAIN':      '#66BB6A',
    'MODERATE STRAIN': '#FFA726',
    'HIGH STRAIN':     '#EF5350',
}
EMOTION_ORDER = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']

# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth(s, w=SMOOTH_WIN):
    return s.rolling(w, min_periods=1, center=True).mean()

def _mmss(x, _pos):
    m, s = divmod(int(max(x, 0)), 60)
    return f'{m}:{s:02}'

def _time_axis(ax, t):
    ax.set_xlim(t.iloc[0], t.iloc[-1])
    ax.xaxis.set_major_formatter(FuncFormatter(_mmss))
    ax.set_xlabel('Time (mm:ss)', fontsize=7)

def _shade(ax, t, labels):
    """Colour the panel background by strain label."""
    prev = t0 = None
    for ti, lbl in zip(t, labels):
        if lbl != prev:
            if prev is not None:
                ax.axvspan(t0, ti, alpha=0.10,
                           color=STRAIN_COLS.get(prev, '#9E9E9E'), lw=0)
            prev, t0 = lbl, ti
    if prev is not None:
        ax.axvspan(t0, t.iloc[-1], alpha=0.10,
                   color=STRAIN_COLS.get(prev, '#9E9E9E'), lw=0)

# ── Panels ────────────────────────────────────────────────────────────────────

def _panel_strain(ax, df):
    t, raw = df['t'], df['strain_score']
    labels = df['strain_label'].fillna('LOW STRAIN')
    _shade(ax, t, labels)
    ax.plot(t, raw,         color='#90CAF9', lw=0.5, alpha=0.55, label='per-frame')
    ax.plot(t, _smooth(raw), color='#1565C0', lw=1.8, label=f'rolling mean ({SMOOTH_WIN} fr)')
    ax.axhline(0.33, color=STRAIN_COLS['MODERATE STRAIN'], lw=0.9, ls='--', alpha=0.8)
    ax.axhline(0.66, color=STRAIN_COLS['HIGH STRAIN'],     lw=0.9, ls='--', alpha=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Strain score', fontsize=8)
    ax.set_title('Cognitive Strain Over Time', fontweight='bold', fontsize=9)
    ax.legend(loc='upper right', fontsize=6.5, frameon=False)
    _time_axis(ax, t)


def _panel_blink(ax, df):
    t = df['t']
    labels = df['strain_label'].fillna('LOW STRAIN')
    _shade(ax, t, labels)
    ax.plot(t, df['blink_rate'], color='#B0BEC5', lw=0.5, alpha=0.6)
    ax.plot(t, _smooth(df['blink_rate']), color='#37474F', lw=1.6)
    ax.set_ylabel('Blinks / min', fontsize=8)
    ax.set_title('Blink Rate', fontsize=9)
    _time_axis(ax, t)


def _panel_jitter(ax, df):
    t = df['t']
    labels = df['strain_label'].fillna('LOW STRAIN')
    _shade(ax, t, labels)
    ax.plot(t, df['gaze_jitter'], color='#CE93D8', lw=0.5, alpha=0.6)
    ax.plot(t, _smooth(df['gaze_jitter']), color='#6A1B9A', lw=1.6)
    ax.axhline(15, color='#AB47BC', lw=0.9, ls='--', alpha=0.85, label='threshold (15 px)')
    ax.set_ylabel('Jitter (px)', fontsize=8)
    ax.set_title('Gaze Jitter', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    _time_axis(ax, t)


def _panel_ear(ax, df):
    t      = df['t']
    ear    = df['ear']
    labels = df['strain_label'].fillna('LOW STRAIN')
    _shade(ax, t, labels)
    ax.plot(t, ear,          color='#80CBC4', lw=0.5, alpha=0.6)
    ax.plot(t, _smooth(ear), color='#00695C', lw=1.6, label='EAR')
    ax.set_ylabel('EAR', fontsize=8)
    ax.set_title('Eye Openness & PERCLOS', fontsize=9)

    # PERCLOS on a secondary y-axis (0–100 %)
    if 'perclos' in df.columns:
        ax2 = ax.twinx()
        p   = df['perclos'] * 100          # → percentage
        ax2.plot(t, p,           color='#EF9A9A', lw=0.5, alpha=0.5)
        ax2.plot(t, _smooth(p),  color='#C62828', lw=1.4, ls='--', label='PERCLOS')
        ax2.axhline(15, color='#C62828', lw=0.8, ls=':', alpha=0.7, label='thr. 15 %')
        ax2.set_ylabel('PERCLOS (%)', fontsize=7, color='#C62828')
        ax2.tick_params(axis='y', labelcolor='#C62828', labelsize=6.5)
        ax2.set_ylim(0, 100)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6.5, frameon=False)
    else:
        ax.legend(fontsize=6.5, frameon=False)

    _time_axis(ax, t)


def _panel_iris(ax, df):
    iris = df['iris_diameter'].dropna()
    if iris.empty:
        ax.text(0.5, 0.5, 'No iris data\n(MediaPipe not detecting)',
                ha='center', va='center', transform=ax.transAxes, fontsize=8, color='grey')
        ax.set_title('Iris Diameter', fontsize=9)
        return
    t_i     = df.loc[iris.index, 't']
    rolling = iris.rolling(SMOOTH_WIN, min_periods=1, center=True).mean()
    _shade(ax, df['t'], df['strain_label'].fillna('LOW STRAIN'))
    ax.scatter(t_i, iris,    s=1.2, color='#FFD54F', alpha=0.45, label='per-frame', rasterized=True)
    ax.plot(t_i, rolling,    color='#E65100', lw=1.6, label='rolling mean')
    ax.set_ylabel('Diameter (px)', fontsize=8)
    ax.set_title('Iris Diameter', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False)
    _time_axis(ax, df['t'])


def _panel_emotion(ax, df):
    counts = (df['emotion']
              .replace('', pd.NA)
              .dropna()
              .value_counts()
              .reindex(EMOTION_ORDER, fill_value=0))
    pct    = counts / counts.sum() * 100
    colors = plt.cm.Set2(np.linspace(0, 1, len(EMOTION_ORDER)))
    bars   = ax.barh(EMOTION_ORDER, pct, color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, pct):
        if val >= 1.0:
            ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=6.5)
    ax.set_xlabel('% of frames', fontsize=7)
    ax.set_xlim(0, max(pct.max() * 1.25, 5))
    ax.set_title('Emotion Distribution', fontsize=9)
    ax.tick_params(axis='y', labelsize=7.5)


def _panel_strain_pie(ax, df):
    labels = list(STRAIN_COLS.keys())
    counts = df['strain_label'].value_counts()
    sizes  = [counts.get(l, 0) for l in labels]
    colors = [STRAIN_COLS[l] for l in labels]
    wedge_kw = dict(width=0.52, edgecolor='white', linewidth=1.2)
    ax.pie(
        sizes, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 1.5 else '',
        pctdistance=0.78, startangle=90, wedgeprops=wedge_kw,
    )
    patches = [Patch(color=STRAIN_COLS[l], label=l.replace(' STRAIN', ''))
               for l in labels]
    ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.22),
              ncol=1, fontsize=7, frameon=False)
    ax.set_title('Strain Breakdown', fontsize=9)

# ── Text summary ──────────────────────────────────────────────────────────────

def _print_summary(df, path):
    dur       = df['t'].iloc[-1]
    mins, sec = divmod(int(dur), 60)
    n         = len(df)
    det_rate  = df['ear'].notna().sum() / n * 100

    sc        = df['strain_score']
    peak_idx  = sc.idxmax()
    peak_t    = df.loc[peak_idx, 't']
    pm, ps    = divmod(int(peak_t), 60)

    label_pct = df['strain_label'].value_counts(normalize=True) * 100
    br        = df['blink_rate']
    ear       = df['ear'].dropna()
    jit       = df['gaze_jitter']
    iris      = df['iris_diameter'].dropna()
    perclos   = df['perclos'].dropna() if 'perclos' in df.columns else pd.Series(dtype=float)
    gaze_pct  = df['gaze_direction'].replace('', pd.NA).dropna().value_counts(normalize=True) * 100
    emo       = df['emotion'].replace('', pd.NA).dropna().value_counts()
    dom_emo   = emo.index[0] if not emo.empty else '—'
    dom_pct   = emo.iloc[0] / emo.sum() * 100 if not emo.empty else 0

    W = 53
    sep = '═' * W
    def row(label, value): print(f'  {label:<22} {value}')

    print(sep)
    print('  COGNITIVE STRAIN SESSION SUMMARY')
    print(f'  {os.path.basename(path)}')
    print(sep)
    row('Duration',           f'{mins} min {sec:02} s')
    row('Frames analysed',    f'{n:,}')
    row('Face detection rate', f'{det_rate:.1f} %')
    print()
    print('  STRAIN OVERVIEW')
    row('Mean score',  f'{sc.mean():.3f}')
    row('Std dev',     f'{sc.std():.3f}')
    row('Peak score',  f'{sc.max():.3f}  (at {pm}:{ps:02})')
    for lbl in ['LOW STRAIN', 'MODERATE STRAIN', 'HIGH STRAIN']:
        pct = label_pct.get(lbl, 0.0)
        bar = '█' * int(pct / 4)
        row(lbl, f'{pct:>5.1f} %  {bar}')
    print()
    print('  BLINK RATE  (blinks / min)')
    row('Mean / Std',  f'{br.mean():.1f} / {br.std():.1f}')
    row('Min  / Max',  f'{br.min():.1f} / {br.max():.1f}')
    print()
    print('  EAR  (Eye Aspect Ratio)')
    row('Mean / Std',  f'{ear.mean():.3f} / {ear.std():.3f}')
    print()
    if not perclos.empty:
        pct_high = (perclos > 0.15).mean() * 100
        print('  PERCLOS  (% of time eye ≥ 80 % closed, 30 s window)')
        row('Mean / Peak',  f'{perclos.mean()*100:.1f} % / {perclos.max()*100:.1f} %')
        row('Time above 15 %', f'{pct_high:.1f} % of session')
        print()
    print('  GAZE JITTER  (px)')
    row('Mean / Peak',  f'{jit.mean():.1f} / {jit.max():.1f}')
    print()
    if not iris.empty:
        print('  IRIS DIAMETER  (px)')
        row('Mean / Std',  f'{iris.mean():.1f} / {iris.std():.1f}')
        print()
    print('  GAZE DIRECTION')
    for direction in ['left', 'center', 'right']:
        pct = gaze_pct.get(direction, 0.0)
        row(direction.capitalize(), f'{pct:.1f} %')
    print()
    row('Dominant emotion', f'{dom_emo}  ({dom_pct:.1f} %)')
    print(sep)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csvs = glob.glob(os.path.join(LOGS_DIR, 'session_*.csv'))
        if not csvs:
            sys.exit(f'No session CSVs found in {os.path.abspath(LOGS_DIR)}')
        csv_path = max(csvs, key=os.path.getmtime)

    print(f'Analysing: {csv_path}')

    df = pd.read_csv(csv_path)
    if df.empty:
        sys.exit('CSV is empty — nothing to analyse.')

    df['t'] = df['timestamp'] - df['timestamp'].iloc[0]
    for col in ('strain_score', 'blink_rate', 'gaze_jitter', 'ear', 'iris_diameter'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.size':          8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.grid':          True,
        'grid.alpha':         0.35,
        'grid.linewidth':     0.5,
    })

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f'Cognitive Strain Analysis  —  {os.path.basename(csv_path)}',
        fontsize=11, fontweight='bold', y=0.995,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

    ax_strain = fig.add_subplot(gs[0, :])
    ax_blink  = fig.add_subplot(gs[1, 0])
    ax_jitter = fig.add_subplot(gs[1, 1])
    ax_ear    = fig.add_subplot(gs[1, 2])
    ax_iris   = fig.add_subplot(gs[2, 0])
    ax_emo    = fig.add_subplot(gs[2, 1])
    ax_pie    = fig.add_subplot(gs[2, 2])

    _panel_strain    (ax_strain, df)
    _panel_blink     (ax_blink,  df)
    _panel_jitter    (ax_jitter, df)
    _panel_ear       (ax_ear,    df)
    _panel_iris      (ax_iris,   df)
    _panel_emotion   (ax_emo,    df)
    _panel_strain_pie(ax_pie,    df)

    # Shared legend for background shading
    patches = [Patch(color=STRAIN_COLS[l], alpha=0.45,
                     label=l.replace(' STRAIN', '').title())
               for l in STRAIN_COLS]
    fig.legend(handles=patches, title='Background shading', title_fontsize=7,
               loc='lower center', ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 0.002))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.splitext(csv_path)[0] + '_analysis.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure saved  →  {out_path}\n')

    _print_summary(df, csv_path)


if __name__ == '__main__':
    main()
