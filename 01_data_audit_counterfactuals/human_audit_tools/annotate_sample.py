import pandas as pd
import os
import sys

# Read a single key without requiring Enter on Windows
try:
    import msvcrt
    def get_key():
        return msvcrt.getch().decode('utf-8').lower()
except ImportError:
    # Fallback for other systems if needed
    import tty
    import termios
    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1).lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    csv_path = 'dataset/v2/human_audit_sample_v9_300.csv'

    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found!")
        return

    df = pd.read_csv(csv_path)

    # Ensure the column exists and replace NaNs with empty strings
    if 'human_verdict' not in df.columns:
        df['human_verdict'] = ''
    df['human_verdict'] = df['human_verdict'].fillna('')

    total = len(df)

    for idx, row in df.iterrows():
        if row['human_verdict'] != '':
            continue  # Already annotated

        clear_screen()
        annotated = sum(df['human_verdict'] != '')
        print(f"--- MANUAL REVIEW: {annotated + 1} / {total} ---")
        print(f"Progress: {annotated/total*100:.1f}%\n")

        print("TEXT:")
        print(f"\"{row['text']}\"\n")

        print(f"Original label: {row['original_label']}")
        print(f"Prompt 1 verdict: {row['prompt1_verdict']}")
        if pd.notna(row['prompt1_reason']) and row['prompt1_reason']:
            print(f"Reason: {row['prompt1_reason']}")

        print("\n" + "="*50)
        print("What is YOUR verdict for this sentence?")
        print("[c] CORRECT (the original label is right)")
        print("[m] MISLABELED (the original label is wrong)")
        print("[w] WEAK_BIAS (weak / ambiguous bias)")
        print("-" * 20)
        print("[s] Skip")
        print("[q] Quit (save and exit)")
        print("="*50)

        while True:
            key = get_key()
            if key in ['c', 'm', 'w', 's', 'q']:
                break

        if key == 'q':
            print("\nSaving and exiting...")
            break
        elif key == 's':
            continue

        # Map the key to the verdict
        verdict_map = {
            'c': 'CORRECT',
            'm': 'MISLABELED',
            'w': 'WEAK_BIAS'
        }

        df.at[idx, 'human_verdict'] = verdict_map[key]

        # Save on every iteration so we don't lose data on error
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"\nReview complete. Annotated {sum(df['human_verdict'] != '')} of {total} sentences.")

if __name__ == '__main__':
    main()
