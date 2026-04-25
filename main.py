"""
Battery SOH & RUL Prediction using LSTM + Transfer Learning
Dataset : CALCE (CS35, CS36, CS37 → pre-train | CS38 → target)
Author  : Student Project, April 2026
"""

# Imports
import os, numpy as np, matplotlib.pyplot as plt, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from pathlib import Path

np.random.seed(42); tf.random.set_seed(42)

# Constants and global state
NOMINAL = 1.1   # nominal capacity in Ah
EOL     = 0.88  # end-of-life threshold (80% of nominal)
WINDOW  = 10   # use last 10 cycles to predict next SOH

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler(feature_range=(0.05, 0.95))

DATA = {}
results = {}
history = None

PLOT_FILES = {
    'Degradation': 'plot1_degradation.png',
    'SOH prediction': 'plot2_soh.png',
    'RUL prediction': 'plot3_rul.png',
    'Comparison': 'plot4_comparison.png',
    'Pre-training loss': 'plot5_loss.png',
}

# Data generation and preprocessing helpers
def generate_battery(n_cycles, seed=0):
    rng = np.random.default_rng(seed)
    c   = np.arange(n_cycles, dtype=float)

    cap = 1.1 * np.exp(-0.0004*c - 1.5e-7*c**2)
    for hump_at, amp in [(int(n_cycles*0.25), 0.012),
                         (int(n_cycles*0.52), 0.008)]:
        cap += amp * np.exp(-((c - hump_at)**2) / 800)

    cap += rng.normal(0, 0.005, n_cycles)
    return np.clip(cap, 0.82, 1.1)


def extract_features(capacity):
    """
    Returns array of shape (n_cycles, 5).
    Features: voltage mean, voltage std, current mean,
              CC charge time, CV charge time.
    """
    rng = np.random.default_rng(1)
    n   = len(capacity)
    soh = capacity / NOMINAL
    F   = np.zeros((n, 5))

    for i, s in enumerate(soh):
        t  = np.linspace(0, 1, 100)
        v  = 3.0 + 1.2*s*(1 - np.exp(-4*t)) + rng.normal(0, 0.007, 100)
        ic = 0.55*s * np.ones(100) + rng.normal(0, 0.003, 100)

        F[i] = [
            np.mean(v),
            np.std(v),
            np.mean(ic),
            0.50 * s + rng.normal(0, 0.01),
            0.35 * s + rng.normal(0, 0.01),
        ]
    return F


def make_sequences(features_norm, soh_norm, window=WINDOW):
    """Sliding window: X shape = (samples, window, n_features)."""
    X, y = [], []
    for i in range(window, len(features_norm)):
        X.append(features_norm[i-window:i])
        y.append(soh_norm[i])
    return np.array(X, np.float32), np.array(y, np.float32)


def prepare_battery(name, train_split):
    """Return train sequences, test sequences, true SOH, train size."""
    d = DATA[name]
    n = len(d['cap'])
    ns = max(WINDOW + 2, int(n * train_split))

    feat_norm = scaler_X.transform(d['features'])
    soh_norm = scaler_Y.transform(d['soh'].reshape(-1, 1)).ravel()

    X_train, y_train = make_sequences(feat_norm[:ns], soh_norm[:ns])
    X_test, _ = make_sequences(feat_norm, soh_norm)
    soh_true = d['soh'][WINDOW:]
    return X_train, y_train, X_test, soh_true, ns


def build_lstm():
    inp = keras.Input(shape=(WINDOW, 5))
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1)(x)
    return Model(inp, out)


def soh_to_rul(soh_array, threshold=0.80):
    """Convert SOH array to RUL array."""
    idx = np.where(soh_array < threshold)[0]
    eol = int(idx[0]) if len(idx) else len(soh_array)
    return np.maximum(eol - np.arange(len(soh_array)), 0).astype(float), eol


def save_figure(fig, filename):
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def initialize_data(log=print):
    log('Generating synthetic CALCE battery data...')
    batteries = {
        'CS35': generate_battery(717, seed=35),
        'CS36': generate_battery(694, seed=36),
        'CS37': generate_battery(758, seed=37),
        'CS38': generate_battery(757, seed=38),
    }

    log('Battery cycles and End-of-Life:')
    for name, cap in batteries.items():
        eol_idx = np.where(cap < EOL)[0]
        eol = int(eol_idx[0]) if len(eol_idx) else len(cap)
        log(f'  {name}: {len(cap)} cycles, EoL @ cycle {eol}')

    DATA.clear()
    for name, cap in batteries.items():
        eol_idx = np.where(cap < EOL)[0]
        DATA[name] = {
            'cap': cap,
            'soh': cap / NOMINAL,
            'features': extract_features(cap),
            'eol': int(eol_idx[0]) if len(eol_idx) else len(cap),
        }

    log('Normalising feature data...')
    src_feats = np.vstack([DATA[b]['features'] for b in ['CS35', 'CS36', 'CS37']])
    src_soh = np.concatenate([DATA[b]['soh'] for b in ['CS35', 'CS36', 'CS37']])
    scaler_X.fit(src_feats)
    scaler_Y.fit(src_soh.reshape(-1, 1))


def fine_tune_predict(split, label, log=print):
    X_train, y_train, X_test, soh_true, ns = prepare_battery('CS38', split)
    m = build_lstm()
    m.load_weights('pretrained.weights.h5')

    m.layers[1].trainable = False
    m.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    m.fit(X_train, y_train, epochs=60, batch_size=16,
          callbacks=[EarlyStopping(patience=8, restore_best_weights=True)], verbose=0)

    for layer in m.layers:
        layer.trainable = True
    m.compile(optimizer=keras.optimizers.Adam(0.0005), loss='mse')
    m.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0)

    pred_norm = m.predict(X_test, verbose=0).ravel()
    pred_soh = scaler_Y.inverse_transform(pred_norm.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(soh_true, pred_soh)
    rmse = np.sqrt(mean_squared_error(soh_true, pred_soh))
    mape = float(np.mean(np.abs(soh_true - pred_soh) / (soh_true + 1e-8)))

    log(f'  {label:15}  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.4f}')
    return pred_soh, soh_true, ns, mae, rmse, mape


def run_experiment(log=print):
    global history, results
    initialize_data(log)

    log('\nPhase A: Pre-training on CS35 + CS36 + CS37 ...')
    X_src = np.vstack([prepare_battery(b, 0.10)[0] for b in ['CS35', 'CS36', 'CS37']])
    y_src = np.concatenate([prepare_battery(b, 0.10)[1] for b in ['CS35', 'CS36', 'CS37']])

    pretrained = build_lstm()
    pretrained.compile(optimizer=keras.optimizers.Adam(0.003), loss='mse')
    history = pretrained.fit(
        X_src, y_src,
        epochs=80, batch_size=32, validation_split=0.15,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0
    )
    pretrained.save_weights('pretrained.weights.h5')
    log(f'  Done. Best val loss = {min(history.history["val_loss"]):.5f}')

    log('\nPhase B: Fine-tuning on CS38 ...')
    results.clear()
    for label, split in [('Early  (10%)', 0.10),
                          ('Mid    (50%)', 0.50),
                          ('Late   (80%)', 0.80)]:
        results[label] = fine_tune_predict(split, label, log=log)

    log('\nBaseline: LSTM from scratch on CS38 (10%) ...')
    X_tr0, y_tr0, X_te0, soh0, _ = prepare_battery('CS38', 0.10)
    scratch = build_lstm()
    scratch.compile('adam', 'mse')
    scratch.fit(X_tr0, y_tr0, epochs=80, batch_size=16,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
    bl_pred = scaler_Y.inverse_transform(scratch.predict(X_te0, verbose=0).reshape(-1, 1)).ravel()
    bl_mae = mean_absolute_error(soh0, bl_pred)
    bl_rmse = np.sqrt(mean_squared_error(soh0, bl_pred))
    log(f"  {'Scratch LSTM':<15}  MAE={bl_mae:.4f}  RMSE={bl_rmse:.4f}")

    pred10, true10, *_ = results['Early  (10%)']
    true_rul, true_eol = soh_to_rul(true10)
    pred_rul, pred_eol = soh_to_rul(pred10)
    log(f'\nRUL: True EoL={true_eol}, Predicted EoL={pred_eol}, Error={abs(true_eol-pred_eol)} cycles')

    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                         'axes.spines.top': False, 'axes.spines.right': False})

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for name, col in zip(DATA, ['#1565C0', '#C62828', '#2E7D32', '#E65100']):
        ax.plot(DATA[name]['cap'], color=col, lw=1.3, label=name)
    ax.axhline(EOL, color='k', ls='--', lw=1, label='EoL (80%)')
    ax.set_xlabel('Cycle'); ax.set_ylabel('Capacity (Ah)')
    ax.set_title('CALCE Battery Degradation', fontweight='bold')
    ax.legend(); plt.tight_layout(); save_figure(fig, PLOT_FILES['Degradation'])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (lbl, res), col in zip(axes, results.items(), ['#C62828', '#2E7D32', '#1565C0']):
        pred, soh_true, ns, mae, rmse, mape = res
        ax.plot(DATA['CS38']['soh'], 'k', lw=1.3, label='True SOH')
        ax.plot(np.arange(WINDOW, WINDOW + len(pred)), pred, color=col,
                lw=1.2, label='LSTM+Transfer')
        ax.axvline(ns, color='navy', ls=':', lw=1.2, label='Split')
        ax.axhline(0.80, color='grey', ls='--', lw=0.8)
        ax.set_title(f'{lbl.strip()}\nMAE={mae:.4f}', fontsize=9, fontweight='bold')
        ax.set_xlabel('Cycle'); ax.set_ylabel('SOH')
        ax.legend(fontsize=7.5); ax.set_ylim(0.70, 1.05)
    plt.suptitle('SOH Prediction on CS38 (Transfer Learning)', fontweight='bold')
    plt.tight_layout(); save_figure(fig, PLOT_FILES['SOH prediction'])

    fig, ax = plt.subplots(figsize=(7, 4))
    cyc = np.arange(WINDOW, WINDOW + len(true_rul))
    ax.plot(cyc, true_rul, 'k', lw=1.5, label='True RUL')
    ax.plot(cyc, pred_rul[:len(true_rul)], '#C62828', lw=1.2, label='Predicted RUL')
    ax.fill_between(cyc, true_rul, pred_rul[:len(true_rul)], alpha=0.12, color='#C62828')
    ax.set_xlabel('Cycle'); ax.set_ylabel('RUL (cycles)')
    ax.set_title(f'RUL Prediction — EoL Error = {abs(true_eol-pred_eol)} cycles', fontweight='bold')
    ax.legend(); plt.tight_layout(); save_figure(fig, PLOT_FILES['RUL prediction'])

    labels = ['LSTM+TL\n10%', 'LSTM+TL\n50%', 'LSTM+TL\n80%', 'Scratch\n10%']
    mae_vals = [results[k][3] for k in results] + [bl_mae]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars = ax.bar(labels, mae_vals, color=['#C62828', '#2E7D32', '#1565C0', '#546E7A'],
                  alpha=0.85, width=0.5)
    for bar, v in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('MAE'); ax.set_title('Model Comparison on CS38', fontweight='bold')
    plt.tight_layout(); save_figure(fig, PLOT_FILES['Comparison'])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(history.history['loss'], '#1565C0', lw=1.3, label='Train')
    ax.plot(history.history['val_loss'], '#C62828', lw=1.3, ls='--', label='Validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('Pre-training Loss (Phase A)', fontweight='bold')
    ax.legend(); plt.tight_layout(); save_figure(fig, PLOT_FILES['Pre-training loss'])

    log('\nAll plots saved!')
    log('\n===== Final Results =====')
    log(f"{'Model':<22} {'MAE':>8} {'RMSE':>8}")
    log('-' * 40)
    for lbl, (_, _, _, mae, rmse, _) in results.items():
        log(f"LSTM+TL {lbl.strip():<14} {mae:.5f}  {rmse:.5f}")
    log(f"{'LSTM Scratch 10%':<22} {bl_mae:.5f}  {bl_rmse:.5f}")


class BatteryUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Battery SOH & RUL UI')
        self.geometry('940x680')
        self.resizable(False, False)
        self.pipeline_running = False
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(fill='x')

        run_button = ttk.Button(control_frame, text='Run Experiment', command=self.start_run)
        run_button.grid(row=0, column=0, padx=4, pady=4, sticky='w')
        self.run_button = run_button

        clear_button = ttk.Button(control_frame, text='Clear Log', command=self.clear_log)
        clear_button.grid(row=0, column=1, padx=4, pady=4, sticky='w')

        for idx, (label, filename) in enumerate(PLOT_FILES.items(), start=2):
            btn = ttk.Button(control_frame, text=label, command=lambda f=filename: self.open_plot(f))
            btn.grid(row=0, column=idx, padx=2, pady=4)

        self.status_var = tk.StringVar(value='Ready')
        status_label = ttk.Label(self, textvariable=self.status_var, anchor='w')
        status_label.pack(fill='x', padx=10)

        self.log_widget = scrolledtext.ScrolledText(self, wrap='word', width=112, height=32, state='disabled')
        self.log_widget.pack(fill='both', expand=True, padx=10, pady=(0, 10))

    def log(self, message):
        self.status_var.set(message if len(message) < 80 else message[:80] + '...')
        self.log_widget.configure(state='normal')
        self.log_widget.insert('end', message + '\n')
        self.log_widget.see('end')
        self.log_widget.configure(state='disabled')

    def clear_log(self):
        self.log_widget.configure(state='normal')
        self.log_widget.delete('1.0', 'end')
        self.log_widget.configure(state='disabled')
        self.status_var.set('Ready')

    def start_run(self):
        if self.pipeline_running:
            return
        self.pipeline_running = True
        self.run_button.configure(state='disabled')
        self.clear_log()
        self.log('Starting experiment...')
        thread = threading.Thread(target=self.run_pipeline, daemon=True)
        thread.start()

    def run_pipeline(self):
        try:
            run_experiment(log=self.log)
            self.log('Experiment completed successfully.')
        except Exception as exc:
            self.log(f'Error: {exc}')
            messagebox.showerror('Error', str(exc))
        finally:
            self.pipeline_running = False
            self.run_button.configure(state='normal')

    def open_plot(self, filename):
        path = Path(filename)
        if path.exists():
            os.startfile(path)
        else:
            self.log(f'Plot file not found: {filename}')


def main():
    app = BatteryUI()
    app.mainloop()


if __name__ == '__main__':
    main()