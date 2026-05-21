import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentence Annotator - Attention Atlas")
        self.root.geometry("1000x700")
        self.root.configure(padx=20, pady=20)

        self.csv_path = 'dataset/v2/human_audit_sample_300.csv'

        if not os.path.exists(self.csv_path):
            messagebox.showerror("Error", f"File not found: {self.csv_path}")
            self.root.destroy()
            return

        self.df = pd.read_csv(self.csv_path)
        if 'human_verdict' not in self.df.columns:
            self.df['human_verdict'] = ''
        self.df['human_verdict'] = self.df['human_verdict'].fillna('')

        self.current_idx = 0

        self.build_ui()
        self.bind_keys()

        if self.find_next_unlabeled():
            self.update_ui()
        else:
            self.show_finished()

    def find_next_unlabeled(self):
        while self.current_idx < len(self.df):
            if self.df.at[self.current_idx, 'human_verdict'] == '':
                return True
            self.current_idx += 1
        return False

    def build_ui(self):
        # Header / Progress
        self.lbl_progress = tk.Label(self.root, text="", font=("Segoe UI", 12, "bold"), fg="#555")
        self.lbl_progress.pack(anchor="w", pady=(0, 10))

        # Text Frame
        frame_text = tk.Frame(self.root, bg="#f8f9fa", bd=1, relief="solid", padx=20, pady=20)
        frame_text.pack(fill="x", pady=(0, 20))

        self.lbl_text = tk.Label(frame_text, text="", font=("Segoe UI", 16), bg="#f8f9fa", wraplength=900, justify="left")
        self.lbl_text.pack(anchor="w")

        # Info Frame
        frame_info = tk.Frame(self.root)
        frame_info.pack(fill="x", pady=(0, 20))

        self.lbl_original = tk.Label(frame_info, text="", font=("Segoe UI", 12))
        self.lbl_original.pack(anchor="w", pady=2)

        self.lbl_prompt_verdict = tk.Label(frame_info, text="", font=("Segoe UI", 12, "bold"), fg="#0056b3")
        self.lbl_prompt_verdict.pack(anchor="w", pady=2)

        self.lbl_prompt_reason = tk.Label(frame_info, text="", font=("Segoe UI", 11, "italic"), fg="#666", wraplength=900, justify="left")
        self.lbl_prompt_reason.pack(anchor="w", pady=2)

        # Instructions / Buttons
        lbl_inst = tk.Label(self.root, text="Press a key or click a button:", font=("Segoe UI", 12, "bold"))
        lbl_inst.pack(pady=(20, 10))

        frame_btns = tk.Frame(self.root)
        frame_btns.pack()

        btn_kwargs = {"font": ("Segoe UI", 11, "bold"), "width": 20, "pady": 10, "cursor": "hand2"}

        btn_c = tk.Button(frame_btns, text="[C] CORRECT\n(label is right)", bg="#d4edda", fg="#155724", command=lambda: self.annotate('CORRECT'), **btn_kwargs)
        btn_c.grid(row=0, column=0, padx=10)

        btn_m = tk.Button(frame_btns, text="[M] MISLABELED\n(label is wrong)", bg="#f8d7da", fg="#721c24", command=lambda: self.annotate('MISLABELED'), **btn_kwargs)
        btn_m.grid(row=0, column=1, padx=10)

        btn_w = tk.Button(frame_btns, text="[W] WEAK_BIAS\n(weak / ambiguous bias)", bg="#fff3cd", fg="#856404", command=lambda: self.annotate('WEAK_BIAS'), **btn_kwargs)
        btn_w.grid(row=0, column=2, padx=10)

        frame_bottom = tk.Frame(self.root)
        frame_bottom.pack(pady=20)

        tk.Button(frame_bottom, text="[S] Skip", width=15, font=("Segoe UI", 10), command=self.skip).grid(row=0, column=0, padx=10)
        tk.Button(frame_bottom, text="[Q] Save and Quit", width=15, font=("Segoe UI", 10), command=self.quit).grid(row=0, column=1, padx=10)

    def bind_keys(self):
        self.root.bind('<c>', lambda e: self.annotate('CORRECT'))
        self.root.bind('<C>', lambda e: self.annotate('CORRECT'))
        self.root.bind('<m>', lambda e: self.annotate('MISLABELED'))
        self.root.bind('<M>', lambda e: self.annotate('MISLABELED'))
        self.root.bind('<w>', lambda e: self.annotate('WEAK_BIAS'))
        self.root.bind('<W>', lambda e: self.annotate('WEAK_BIAS'))
        self.root.bind('<s>', lambda e: self.skip())
        self.root.bind('<S>', lambda e: self.skip())
        self.root.bind('<q>', lambda e: self.quit())
        self.root.bind('<Q>', lambda e: self.quit())

    def update_ui(self):
        row = self.df.iloc[self.current_idx]
        total = len(self.df)
        annotated = sum(self.df['human_verdict'] != '')

        self.lbl_progress.config(text=f"Progress: {annotated} of {total} annotated ({(annotated/total)*100:.1f}%)")
        self.lbl_text.config(text=f'"{row["text"]}"')

        self.lbl_original.config(text=f"Original label: {row['original_label']}")
        self.lbl_prompt_verdict.config(text=f"Automatic verdict: {row['prompt1_verdict']}")

        reason = row['prompt1_reason']
        if pd.notna(reason) and str(reason).strip():
            self.lbl_prompt_reason.config(text=f"Model reason: {reason}")
        else:
            self.lbl_prompt_reason.config(text="")

    def annotate(self, verdict):
        self.df.at[self.current_idx, 'human_verdict'] = verdict
        self.df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')

        self.current_idx += 1
        if self.find_next_unlabeled():
            self.update_ui()
        else:
            self.show_finished()

    def skip(self):
        self.current_idx += 1
        if self.find_next_unlabeled():
            self.update_ui()
        else:
            self.show_finished()

    def quit(self):
        self.root.destroy()

    def show_finished(self):
        self.lbl_text.config(text="All sentences in the sample have been annotated.")
        self.lbl_original.config(text="")
        self.lbl_prompt_verdict.config(text="")
        self.lbl_prompt_reason.config(text="")
        messagebox.showinfo("Done", "All sentences have been successfully annotated.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotatorApp(root)
    root.mainloop()
