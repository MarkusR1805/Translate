#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helsinki-NLP Translator GUI
PySide6 + transformers | Dark Theme | macOS-optimiert
"""

import sys
import os
import warnings
import logging
import torch
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QTextEdit, QComboBox, QPushButton,
                               QLabel, QStatusBar, QMessageBox)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QClipboard, QFont, QColor, QPalette
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────────────────────────────────────────
# Logging & Warnings
# ─────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────
# Modell-Konfiguration (ohne Chinesisch)
# ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "Englisch → Deutsch": ["Helsinki-NLP/opus-mt-en-de"],
    "Deutsch → Englisch": ["Helsinki-NLP/opus-mt-de-en"],
    "Französisch → Deutsch": ["Helsinki-NLP/opus-mt-fr-de"],
    "Deutsch → Französisch": ["Helsinki-NLP/opus-mt-de-fr"],
}


class TranslationWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, text: str, lang_key: str, cache_dir: str):
        super().__init__()
        self.text = text
        self.lang_key = lang_key
        self.cache_dir = cache_dir

    def run(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.status.emit(f"Gerät: {device.type.upper()}")

            current_text = self.text
            model_names = MODEL_CONFIGS[self.lang_key]
            hf_token = os.getenv("HF_TOKEN")

            for i, model_name in enumerate(model_names, 1):
                model_short = model_name.split("/")[-1]
                self.status.emit(f"Schritt {i}/{len(model_names)}: Lade '{model_short}'...")

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, cache_dir=self.cache_dir, token=hf_token
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, cache_dir=self.cache_dir, token=hf_token
                )
                model.to(device)
                model.eval()

                self.status.emit(f"Schritt {i}/{len(model_names)}: Übersetze...")
                inputs = tokenizer(current_text, return_tensors="pt", truncation=True, max_length=512).to(device)
                outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                current_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                del model, tokenizer, inputs, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.finished.emit(current_text)

        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🌐 Helsinki-NLP Übersetzer")
        self.setFixedSize(1100, 750)
        self.center_on_screen()
        self.setup_dark_theme()

        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._setup_ui()
        self.worker = None

    def center_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)

    def setup_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#1e2329"))
        palette.setColor(QPalette.WindowText, QColor("#e6e6e6"))
        palette.setColor(QPalette.Base, QColor("#2a2f35"))
        palette.setColor(QPalette.AlternateBase, QColor("#353a40"))
        palette.setColor(QPalette.Text, QColor("#e6e6e6"))
        palette.setColor(QPalette.Button, QColor("#3a4047"))
        palette.setColor(QPalette.ButtonText, QColor("#e6e6e6"))
        palette.setColor(QPalette.Highlight, QColor("#5294e2"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        QApplication.setPalette(palette)

        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #1e2329;
                color: #e6e6e6;
                font-family: Roboto Mono, sans-serif;
            }
            QTextEdit {
                background: #2a2f35;
                border: 1px solid #444;
                border-radius: 8px;
                padding: 10px;
                font-size: 16pt;
                line-height: 1.4;
            }
            QTextEdit:disabled {
                background: #252a30;
                color: #aaa;
            }
            QComboBox {
                background: #3a4047;
                border: 1px solid #444;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 14pt;
                min-width: 280px;
            }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #e6e6e6;
                margin-right: 8px;
            }
            QPushButton {
                background: #3a4047;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14pt;
                font-weight: 500;
            }
            QPushButton:hover { background: #4a5057; }
            QPushButton:pressed { background: #2a3037; }
            QPushButton:disabled { background: #2a2f35; color: #666; }
            QLabel { font-size: 14pt; color: #ccc; }
            QStatusBar {
                background: #1a1e22;
                color: #aaa;
                font-size: 12pt;
                border-top: 1px solid #333;
            }
        """)

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ─── Sprachauswahl ─────────────────────────────────
        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel("<b>Sprachrichtung:</b>"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(MODEL_CONFIGS.keys())
        lang_layout.addWidget(self.lang_combo)
        lang_layout.addStretch()
        main_layout.addLayout(lang_layout)

        # ─── HAUPTBEREICH: Zwei Spalten mit Textfeld + Button ───
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # LINKE SPALTE: Eingabe + "Aus Zwischenablage" Button
        left_column = QVBoxLayout()
        left_column.addWidget(QLabel("Eingabe:"))

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Text hier eingeben oder einfügen…")
        left_column.addWidget(self.input_text)

        self.paste_btn = QPushButton("📋 Aus Zwischenablage")
        self.paste_btn.setFixedHeight(42)
        self.paste_btn.setToolTip("Text aus der System-Zwischenablage einfügen")
        left_column.addWidget(self.paste_btn)

        content_layout.addLayout(left_column)

        # RECHTE SPALTE: Übersetzung + "In Zwischenablage" Button
        right_column = QVBoxLayout()
        right_column.addWidget(QLabel("Übersetzung:"))

        self.output_text = QTextEdit()
        self.output_text.setPlaceholderText("Ergebnis erscheint hier…")
        self.output_text.setReadOnly(True)
        right_column.addWidget(self.output_text)

        self.copy_btn = QPushButton("📋 In Zwischenablage")
        self.copy_btn.setFixedHeight(42)
        self.copy_btn.setToolTip("Übersetzung in die System-Zwischenablage kopieren")
        right_column.addWidget(self.copy_btn)

        content_layout.addLayout(right_column)

        main_layout.addLayout(content_layout)

        # ─── UNTERE LEISTE: Übersetzen & Beenden ─────────
        bottom_layout = QHBoxLayout()

        self.translate_btn = QPushButton("🌐 Übersetzen")
        self.translate_btn.setStyleSheet("""
            font-weight: bold;
            background: #5294e2;
            color: white;
            font-size: 15pt;
        """)
        self.translate_btn.setFixedHeight(48)
        self.translate_btn.setFixedWidth(200)

        self.exit_btn = QPushButton("❌ Beenden")
        self.exit_btn.setFixedHeight(42)
        self.exit_btn.setFixedWidth(150)

        bottom_layout.addWidget(self.translate_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.exit_btn)

        main_layout.addLayout(bottom_layout)

        # ─── Statusleiste ──────────────────────────────────
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Bereit.")

        # ─── Signale verbinden ─────────────────────────────
        self.paste_btn.clicked.connect(self.paste_from_clipboard)
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.translate_btn.clicked.connect(self.start_translation)
        self.exit_btn.clicked.connect(self.close)

    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if text.strip():
            self.input_text.setPlainText(text)
            self.statusBar.showMessage("✅ Aus Zwischenablage eingefügt", 2000)
        else:
            self.statusBar.showMessage("⚠️ Zwischenablage ist leer", 2000)

    def copy_to_clipboard(self):
        output = self.output_text.toPlainText()
        if output.strip():
            QApplication.clipboard().setText(output)
            self.statusBar.showMessage("✅ In Zwischenablage kopiert", 2000)
        else:
            self.statusBar.showMessage("⚠️ Nichts zum Kopieren", 2000)

    def start_translation(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Eingabe leer", "Bitte gib einen Text ein.")
            return

        lang_key = self.lang_combo.currentText()
        self.translate_btn.setEnabled(False)
        self.output_text.clear()
        self.statusBar.showMessage("⏳ Übersetze...")

        self.worker = TranslationWorker(text, lang_key, self.cache_dir)
        self.worker.status.connect(lambda msg: self.statusBar.showMessage(msg))
        self.worker.finished.connect(self.on_translation_finished)
        self.worker.error.connect(self.on_translation_error)
        self.worker.start()

    def on_translation_finished(self, result: str):
        self.output_text.setPlainText(result)
        self.statusBar.showMessage("✅ Übersetzung abgeschlossen!", 4000)
        self.translate_btn.setEnabled(True)

    def on_translation_error(self, error_msg: str):
        self.statusBar.showMessage("❌ Fehler aufgetreten", 6000)
        self.translate_btn.setEnabled(True)
        QMessageBox.critical(self, "Fehler", f"Ein Fehler ist aufgetreten:\n\n{error_msg}")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    if sys.platform == "darwin":
        QApplication.setApplicationName("Helsinki Translator")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    # macOS: App in den Vordergrund holen
    if sys.platform == "darwin":
        import subprocess
        subprocess.run(["osascript", "-e", 'tell application "System Events" to set frontmost of process "Python" to true'],
                      capture_output=True)

    sys.exit(app.exec())
