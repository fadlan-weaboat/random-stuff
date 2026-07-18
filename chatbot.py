"""
Chatbot with SQLite-backed chat history and tool use, running on Mistral 7B.
Picks the best available backend in this order: XPU (Intel oneAPI) -> CUDA
(NVIDIA) -> CPU, with automatic fallback to CPU at runtime if the chosen
GPU backend fails mid-generation (e.g. Triton compilation error on XPU, or
an out-of-memory error on CUDA).

On Windows with an Intel XPU, launch via run_chatbot.bat, which sets up the
oneAPI/icx.exe environment XPU+Triton needs. On CUDA or CPU-only machines
(no oneAPI installed), the oneAPI-specific setup below is skipped
automatically and you can just run `python chatbot.py` directly.
"""

import os

# Extension-module dependency dirs for Triton's compiled XPU .pyd
# (sycl8.dll, ur_win_proxy_loader.dll, vcruntime140.dll, etc). Only relevant
# on Windows machines with Intel oneAPI installed - silently skipped
# elsewhere (e.g. a Linux CUDA box), since none of these paths will exist.
if hasattr(os, 'add_dll_directory'):
    for _dll_dir in [
        r'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin',
        r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Redist\MSVC\14.51.36231\x64\Microsoft.VC145.CRT',
    ]:
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except OSError:
                pass

import torch
torch._dynamo.config.disable = True  # keep torch.compile out of the picture entirely

from transformers import AutoModelForCausalLM, AutoTokenizer
import sqlite3
import warnings
import json
import re
import secrets
import string

def _gen_tool_call_id():
    """Mistral's chat template requires a 9-char alphanumeric tool_call_id."""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(9))

# Fallback in case run_chatbot.bat's oneAPI setvars wasn't sourced. Only
# applied if the icx.exe path actually exists on this machine - otherwise
# CC/CXX are left alone, so this has no effect on a CUDA or CPU-only setup
# (Triton compilation is XPU-only here; torch.compile is disabled above).
_ICX_PATH = r'C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin\icx.exe'
if os.path.exists(_ICX_PATH):
    if not os.environ.get('CXX') or not os.path.exists(os.environ.get('CXX', '')):
        os.environ['CXX'] = _ICX_PATH
    if not os.environ.get('CC') or not os.path.exists(os.environ.get('CC', '')):
        os.environ['CC'] = _ICX_PATH

warnings.filterwarnings("ignore", message=r".*_check_is_size.*")

FORCE_CPU_ONLY = False
_force_cpu_mode = False  # flips True at runtime if the GPU backend (XPU or CUDA) fails once

# ======================
# DATABASE
# ======================

# The four allowed values for master.user_type. Enforced both in Python
# (get_or_create_user, the startup prompt) and in the DB itself via a
# CHECK constraint, so bad values can't sneak in through some other path.
USER_TYPES = ("government", "vendor", "school", "other")

def _master_has_type_constraint(conn):
    """True if the existing master table's CREATE statement already has
    the user_type CHECK constraint (i.e. no migration needed)."""
    c = conn.cursor()
    c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='master'")
    row = c.fetchone()
    if row is None:
        return True  # table doesn't exist yet, nothing to migrate
    sql = row[0] or ""
    return "user_type" in sql and "CHECK" in sql.upper()

def _migrate_master_user_types(conn):
    """Upgrade an older master table (no CHECK constraint, possibly holding
    invalid/blank user_type values like '' or 'demo') to the new schema.
    Any existing value that isn't one of USER_TYPES is mapped to 'other'
    rather than dropped, so no rows or chat history are lost."""
    c = conn.cursor()
    print("[MIGRATE] Upgrading 'master' table to enforce the 4 user_type values...")

    c.execute("ALTER TABLE master RENAME TO master_old")
    c.execute('''
        CREATE TABLE master (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            region TEXT,
            user_type TEXT NOT NULL DEFAULT 'other' CHECK (user_type IN ('government', 'vendor', 'school', 'other')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute("SELECT user_id, user_name, region, user_type, created_at FROM master_old")
    old_rows = c.fetchall()
    for user_id, user_name, region, user_type, created_at in old_rows:
        normalized_type = user_type if user_type in USER_TYPES else "other"
        c.execute(
            "INSERT INTO master (user_id, user_name, region, user_type, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_name, region, normalized_type, created_at)
        )

    # Keep AUTOINCREMENT counting up from where it left off rather than
    # reusing old user_ids (which chats.user_id may still reference).
    c.execute("SELECT MAX(user_id) FROM master")
    max_id = c.fetchone()[0] or 0
    c.execute("INSERT OR REPLACE INTO sqlite_sequence (name, seq) VALUES ('master', ?)", (max_id,))

    c.execute("DROP TABLE master_old")
    conn.commit()
    print(f"[MIGRATE] Done - {len(old_rows)} user(s) migrated (non-matching types set to 'other').")

def init_database():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS master (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT NOT NULL,
            region TEXT,
            user_type TEXT NOT NULL DEFAULT 'other' CHECK (user_type IN ('government', 'vendor', 'school', 'other')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES master(user_id)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)')
    conn.commit()

    if not _master_has_type_constraint(conn):
        _migrate_master_user_types(conn)

    ensure_vendor_documents_table(conn)

    conn.close()

def get_or_create_user(user_name, region="unknown", user_type="other"):
    user_type = (user_type or "other").strip().lower()
    if user_type not in USER_TYPES:
        raise ValueError(f"Invalid user_type {user_type!r}; must be one of {USER_TYPES}")

    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "SELECT user_id FROM master WHERE user_name = ? AND region = ? AND user_type = ?",
        (user_name, region, user_type)
    )
    result = c.fetchone()
    if result:
        user_id = result[0]
    else:
        c.execute(
            "INSERT INTO master (user_name, region, user_type) VALUES (?, ?, ?)",
            (user_name, region, user_type)
        )
        user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id

def save_chat_message(user_id, message, response):
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (user_id, message, response) VALUES (?, ?, ?)",
        (user_id, message, response)
    )
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=50):
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "SELECT message, response, timestamp FROM chats WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    history = c.fetchall()
    conn.close()
    return history

def get_user_stats(user_id):
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "SELECT user_name, region, user_type, created_at FROM master WHERE user_id = ?",
        (user_id,)
    )
    user_info = c.fetchone()
    c.execute("SELECT COUNT(*) FROM chats WHERE user_id = ?", (user_id,))
    chat_count = c.fetchone()[0]
    c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM chats WHERE user_id = ?", (user_id,))
    time_range = c.fetchone()
    conn.close()
    return {
        'user_info': user_info,
        'chat_count': chat_count,
        'first_chat': time_range[0] if time_range[0] else None,
        'last_chat': time_range[1] if time_range[1] else None,
    }

# ======================
# VENDOR DOCUMENTS (single shared table, FK'd to master - same pattern as chats)
# ======================

def _legacy_vendor_tables(conn):
    """Find any per-vendor tables from the old one-table-per-vendor design
    (vendor_docs_<id>), so they can be folded into the new shared table."""
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vendor\\_docs\\_%' ESCAPE '\\'")
    return [r[0] for r in c.fetchall()]

def _migrate_legacy_vendor_tables(conn):
    """One-time migration: copy rows out of each old vendor_docs_<id> table
    into vendor_documents (setting vendor_id from the table name), then drop
    the old table. Keeping one table per vendor didn't scale, so this
    consolidates everything into vendor_documents, mirroring how chats
    references master via a foreign key instead of per-user tables."""
    legacy_tables = _legacy_vendor_tables(conn)
    if not legacy_tables:
        return
    c = conn.cursor()
    total = 0
    for table in legacy_tables:
        vendor_id = int(table.rsplit("_", 1)[-1])
        c.execute(f'SELECT doc_name, doc_type, content, notes, uploaded_at FROM "{table}"')
        rows = c.fetchall()
        for doc_name, doc_type, content, notes, uploaded_at in rows:
            c.execute(
                "INSERT INTO vendor_documents (vendor_id, doc_name, doc_type, content, notes, uploaded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (vendor_id, doc_name, doc_type, content, notes, uploaded_at)
            )
        c.execute(f'DROP TABLE "{table}"')
        total += len(rows)
    conn.commit()
    print(f"[MIGRATE] Consolidated {len(legacy_tables)} legacy vendor table(s), {total} document(s), into 'vendor_documents'.")

def ensure_vendor_documents_table(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS vendor_documents (
            doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_id INTEGER NOT NULL,
            doc_name TEXT NOT NULL,
            doc_type TEXT,
            content BLOB NOT NULL,
            notes TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (vendor_id) REFERENCES master(user_id)
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_vendor_documents_vendor_id ON vendor_documents(vendor_id)')
    conn.commit()
    _migrate_legacy_vendor_tables(conn)

def _is_vendor(conn, user_id):
    c = conn.cursor()
    c.execute("SELECT user_type FROM master WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    return row is not None and row[0] == "vendor"

def add_vendor_document(vendor_id, doc_name, content, doc_type=None, notes=None):
    """Store a document for a vendor. `content` may be str (encoded as
    utf-8 text) or bytes (stored as-is, e.g. a PDF/image read from disk)."""
    if isinstance(content, str):
        content_blob = content.encode('utf-8')
        doc_type = doc_type or 'text'
    elif isinstance(content, (bytes, bytearray)):
        content_blob = bytes(content)
    else:
        raise TypeError(f"content must be str or bytes, got {type(content).__name__}")

    conn = sqlite3.connect('chatbot.db')
    if not _is_vendor(conn, vendor_id):
        conn.close()
        raise ValueError(f"user_id={vendor_id} is not a vendor (or doesn't exist)")

    c = conn.cursor()
    c.execute(
        "INSERT INTO vendor_documents (vendor_id, doc_name, doc_type, content, notes) VALUES (?, ?, ?, ?, ?)",
        (vendor_id, doc_name, doc_type, content_blob, notes)
    )
    conn.commit()
    doc_id = c.lastrowid
    conn.close()
    return doc_id

def get_vendor_documents(vendor_id, limit=50):
    """List a vendor's documents (metadata only - not the content blob,
    to keep results small)."""
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "SELECT doc_id, doc_name, doc_type, notes, uploaded_at, length(content) "
        "FROM vendor_documents WHERE vendor_id = ? ORDER BY uploaded_at DESC LIMIT ?",
        (vendor_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return rows

def get_vendor_document(vendor_id, doc_id):
    """Fetch one document's full content (as bytes) plus its metadata.
    Scoped to vendor_id so one vendor can't pull another's doc by guessing an ID."""
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "SELECT doc_name, doc_type, content, notes, uploaded_at FROM vendor_documents "
        "WHERE vendor_id = ? AND doc_id = ?",
        (vendor_id, doc_id)
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    doc_name, doc_type, content, notes, uploaded_at = row
    return {"doc_name": doc_name, "doc_type": doc_type, "content": content, "notes": notes, "uploaded_at": uploaded_at}

# ======================
# TOOLS
# ======================
# NOTE: these functions themselves still look up any user by ID with no
# restriction - the access-control check (government users limited to their
# own user_id) is enforced one layer up, in generate_response_with_tools'
# dispatch loop, before these are ever called. Other user types (vendor,
# school, other) are unrestricted here; add checks for them too before
# exposing this to untrusted end users.

def tool_get_user_by_id(user_id):
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return {"error": f"Invalid user_id: {user_id!r} (must be an integer)"}

    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute(
        "SELECT user_id, user_name, region, user_type, created_at FROM master WHERE user_id = ?",
        (user_id,)
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return {"error": f"No user found with user_id={user_id}"}
    return {"user_id": row[0], "user_name": row[1], "region": row[2], "user_type": row[3], "created_at": row[4]}

def tool_get_user_chats(user_id, limit=10):
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        return {"error": f"Invalid user_id: {user_id!r} (must be an integer)"}
    try:
        limit = int(limit) if limit is not None else 10
    except (TypeError, ValueError):
        limit = 10
    limit = max(1, min(limit, 100))

    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute("SELECT 1 FROM master WHERE user_id = ?", (user_id,))
    if c.fetchone() is None:
        conn.close()
        return {"error": f"No user found with user_id={user_id}"}

    c.execute(
        "SELECT message, response, timestamp FROM chats WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return {
        "user_id": user_id,
        "chat_count_returned": len(rows),
        "chats": [{"message": m, "response": r, "timestamp": t} for (m, r, t) in rows],
    }

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_by_id",
            "description": "Look up a user's profile information (name, region, type, when they joined) by their numeric user ID.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer", "description": "The numeric ID of the user to look up."}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_chats",
            "description": "Retrieve a user's recent chat history (their messages and the bot's responses) by their numeric user ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "The numeric ID of the user whose chat history to retrieve."},
                    "limit": {"type": "integer", "description": "Maximum number of recent chat entries to return (default 10, max 100)."},
                },
                "required": ["user_id"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "get_user_by_id": tool_get_user_by_id,
    "get_user_chats": tool_get_user_chats,
}

def _find_bracketed_json_candidates(text):
    """Find '[' ... ']' spans (bracket-depth aware) for JSON embedded in prose."""
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start:i + 1])
                    start = None
    return candidates

def _lenient_json_loads(candidate):
    """json.loads with a repair pass for unquoted placeholder values."""
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = re.sub(r':\s*([A-Za-z_][A-Za-z0-9_]*)\s*([,}\]])', r': "\1"\2', candidate)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        return None

def _extract_tool_calls(text):
    """Best-effort extraction of tool calls from raw model output text."""
    candidates = []

    marker_match = re.search(r"\[TOOL_CALLS\]\s*(\[.*\]|\{.*\})", text, re.DOTALL)
    if marker_match:
        candidates.append(marker_match.group(1))

    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        candidates.append(m.group(1))

    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        candidates.append(stripped)

    candidates.extend(_find_bracketed_json_candidates(text))

    for candidate in candidates:
        parsed = _lenient_json_loads(candidate)
        if parsed is None:
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            continue

        calls = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("function", {}).get("name")
            arguments = item.get("arguments") or item.get("function", {}).get("arguments") or {}
            if isinstance(arguments, str):
                parsed_args = _lenient_json_loads(arguments)
                arguments = parsed_args if isinstance(parsed_args, dict) else {}
            if name:
                calls.append({"name": name, "arguments": arguments})
        if calls:
            return calls

    return []

# ======================
# MODEL
# ======================
def _detect_device():
    """Pick the best available backend: XPU (Intel) -> CUDA (NVIDIA) -> CPU."""
    if FORCE_CPU_ONLY:
        return torch.device("cpu"), "CPU (forced)"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu"), "XPU"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "CPU"

def setup_model():
    print("\n[SETUP] Setting up AI model...")

    device, device_label = _detect_device()
    print(f"[OK] Using device: {device_label}")

    model_path = "./mistral-7b-v0.3-int4-inc"
    if not os.path.exists(model_path):
        print(f"[WARN] Local model path not found ({model_path}), using hosted instruct checkpoint")
        model_path = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    uses_accelerate_device_map = str(device).startswith(("xpu", "cuda"))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if uses_accelerate_device_map else None,
            attn_implementation="eager",
        )
        print("[OK] Model loaded (bfloat16)")
    except Exception as e:
        print(f"[WARN] bfloat16 load failed ({e}), falling back to default loading")
        model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager")
        print("[OK] Model loaded (default settings)")

    if hasattr(model, 'to') and str(device) != 'cpu':
        try:
            model = model.to(device)
            print(f"[OK] Model moved to {device}")
        except Exception as e:
            print(f"[WARN] Could not move model to {device}: {e}")

    model.eval()
    return model, tokenizer, device

def _run_generate(model, tokenizer, device, input_ids, attention_mask, max_new_tokens):
    generate_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        cache_implementation="dynamic",
    )
    with torch.no_grad():
        try:
            outputs = model.generate(**generate_kwargs, disable_compile=True)
        except TypeError:
            outputs = model.generate(**generate_kwargs)
    new_tokens = outputs[0][input_ids.shape[-1]:]
    # skip_special_tokens=False here on purpose - [TOOL_CALLS] is a special
    # token, and stripping it would make tool calls undetectable.
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    clean_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return raw_text, clean_text

RESTRICTED_LOOKUP_TOOLS = {"get_user_by_id", "get_user_chats"}  # both take a user_id argument

def generate_response_with_tools(model, tokenizer, device, prompt, current_user_id=None, current_user_type=None, max_new_tokens=300, max_tool_iterations=3):
    """Generate a response, letting the model call get_user_by_id / get_user_chats first if needed.

    Access control: government users have unrestricted lookup access (can
    look up any user_id). Every other user type (vendor, school, other) may
    only look up their own record/chats - the model is told this in the
    system prompt, but the real enforcement happens below in the
    tool-dispatch loop, since the model's tool-call arguments can't be
    trusted on their own."""
    global _force_cpu_mode

    if _force_cpu_mode and str(device).startswith(("xpu", "cuda")):
        model.to('cpu')
        device = torch.device('cpu')

    id_hint = (
        f" The person you are currently talking to has user_id={current_user_id}. "
        f"When they say 'my', 'our', or 'me', use {current_user_id} as the user_id "
        f"argument - never use a placeholder like YOUR_USER_ID."
        if current_user_id is not None else ""
    )
    if current_user_type == "government":
        id_hint += " This person is a government-type user and may look up any user_id."
    elif current_user_type is not None:
        id_hint += (
            f" This person is a {current_user_type}-type user, so get_user_by_id and "
            f"get_user_chats will only succeed for user_id={current_user_id} "
            f"(their own record) - do not attempt to look up any other user_id."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools for looking up "
                "user records in a database. Use get_user_by_id to look up a "
                "user's profile, and get_user_chats to see a user's past "
                "messages. Only call a tool when the person actually asks about "
                "a specific user or their chat history - otherwise just answer "
                "normally." + id_hint
            ),
        },
        {"role": "user", "content": prompt},
    ]

    for _ in range(max_tool_iterations):
        try:
            templated = tokenizer.apply_chat_template(
                messages, tools=TOOLS, add_generation_prompt=True, return_tensors="pt",
            )
        except Exception as e:
            # Keep the full conversation (system prompt + any tool results already
            # gathered) - just drop the tool schema, rather than discarding
            # everything and reverting to the bare original prompt.
            print(f"[WARN] Tool-aware chat template failed ({e}), retrying without tool schema")
            try:
                templated = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors="pt",
                )
            except Exception as e2:
                print(f"[WARN] Chat template still failed ({e2}), falling back to plain prompt")
                templated = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt",
                )

        if hasattr(templated, "input_ids"):
            input_ids = templated.input_ids.to(device)
            attention_mask = templated.get("attention_mask")
        elif isinstance(templated, dict):
            input_ids = templated["input_ids"].to(device)
            attention_mask = templated.get("attention_mask")
        else:
            input_ids = templated.to(device)
            attention_mask = None
        attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids)

        try:
            raw_text, clean_text = _run_generate(model, tokenizer, device, input_ids, attention_mask, max_new_tokens)
        except Exception as e:
            error_msg = str(e)
            is_xpu_triton_failure = (
                str(device).startswith("xpu")
                and ("compiler" in error_msg.lower() or "triton" in error_msg.lower())
            )
            _cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
            is_cuda_oom = str(device).startswith("cuda") and (
                "out of memory" in error_msg.lower()
                or (_cuda_oom_type is not None and isinstance(e, _cuda_oom_type))
            )
            if is_xpu_triton_failure or is_cuda_oom:
                reason = "Triton compilation failed on XPU" if is_xpu_triton_failure else "Out of memory on CUDA"
                print(f"[WARN] {reason}, falling back to CPU for this request")
                _force_cpu_mode = True
                model.to('cpu')
                if is_cuda_oom:
                    torch.cuda.empty_cache()
                device = torch.device('cpu')
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                raw_text, clean_text = _run_generate(model, tokenizer, device, input_ids, attention_mask, max_new_tokens)
            else:
                raise

        calls = _extract_tool_calls(raw_text)
        if not calls:
            return clean_text if clean_text else "I'm not sure how to respond to that."

        for c in calls:
            c["id"] = _gen_tool_call_id()

        messages.append({"role": "assistant", "content": clean_text or None, "tool_calls": [
            {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls
        ]})

        for call in calls:
            name = call["name"]
            arguments = call.get("arguments", {})
            func = TOOL_DISPATCH.get(name)

            denied = False
            if current_user_type != "government" and name in RESTRICTED_LOOKUP_TOOLS:
                requested_id = arguments.get("user_id")
                try:
                    denied = current_user_id is None or int(requested_id) != int(current_user_id)
                except (TypeError, ValueError):
                    denied = True  # non-numeric/missing user_id - deny rather than guess

            if denied:
                result = {"error": "Access denied: you may only look up your own records."}
            elif func is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = func(**arguments)
                except TypeError as e:
                    result = {"error": f"Invalid arguments for {name}: {e}"}
                except Exception as e:
                    result = {"error": f"Tool '{name}' raised an error: {e}"}

            print(f"[TOOL] {name}({arguments}) -> {result}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": json.dumps(result)})

    return "I looked that up but had trouble putting together a final answer - could you rephrase your question?"

def prompt_for_user_type():
    """Ask the person to pick one of the 4 allowed user types from a
    numbered menu. Re-prompts on anything else - typos, blank input,
    numbers out of range - instead of silently defaulting."""
    print("\nSelect user type:")
    for i, ut in enumerate(USER_TYPES, 1):
        print(f"  {i}. {ut}")

    while True:
        choice = input("Enter number (1-4) or type name: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(USER_TYPES):
            return USER_TYPES[int(choice) - 1]
        if choice in USER_TYPES:
            return choice
        print(f"  Not a valid choice. Please enter 1-{len(USER_TYPES)} or one of {', '.join(USER_TYPES)}.")

def select_user():
    """Let the person either log in with an existing user_id or create a new
    user. Returns (user_id, user_name, user_type)."""
    print("\n1. Log in with an existing user ID")
    print("2. Create a new user")

    while True:
        choice = input("Choose (1/2): ").strip()

        if choice == "1":
            raw_id = input("Enter your user ID: ").strip()
            info = tool_get_user_by_id(raw_id)
            if "error" in info:
                print(f"   [ERROR] {info['error']}")
                continue  # back to the 1/2 menu
            print(f"[OK] Logged in as '{info['user_name']}' ({info['user_type']}), ID: {info['user_id']}")
            return info["user_id"], info["user_name"], info["user_type"]

        elif choice == "2":
            user_name = input("Enter your name: ").strip() or "DemoUser"
            region = input("Enter your region (optional): ").strip() or "unknown"
            user_type = prompt_for_user_type()
            user_id = get_or_create_user(user_name, region, user_type)
            print(f"[OK] User '{user_name}' ({user_type}) initialized with ID: {user_id}")
            return user_id, user_name, user_type

        else:
            print("   Please enter 1 or 2.")

def get_demo_response(prompt):
    """Fallback canned responses if the model failed to load at all."""
    prompt_lower = prompt.lower().strip()
    if any(g in prompt_lower for g in ['hello', 'hi', 'hey']):
        return "Hello! I'm your AI assistant. How can I help you today?"
    elif 'how are you' in prompt_lower:
        return "I'm functioning well, thank you! How are you?"
    elif 'your name' in prompt_lower:
        return "I'm an AI assistant powered by Mistral 7B."
    elif any(f in prompt_lower for f in ['bye', 'goodbye']):
        return "Goodbye! Have a wonderful day!"
    else:
        return f"I understand you said: '{prompt}'. I'm currently running in fallback mode, but I'm still here to chat with you!"

# ======================
# MAIN
# ======================
def _handle_docs_command(user_id):
    docs = get_vendor_documents(user_id, limit=20)
    if not docs:
        print("   No documents on file yet. Use 'adddoc' to add one.")
        return
    for doc_id, doc_name, doc_type, notes, uploaded_at, size_bytes in docs:
        notes_part = f" - {notes}" if notes else ""
        print(f"  [{doc_id}] {doc_name} ({doc_type or 'unknown'}, {size_bytes} bytes) - {uploaded_at}{notes_part}")

def _handle_adddoc_command(user_id):
    path = input("   File path to upload (or leave blank to paste text instead): ").strip()
    notes = input("   Notes/description (optional): ").strip() or None

    if path:
        if not os.path.isfile(path):
            print(f"   [ERROR] File not found: {path}")
            return
        with open(path, "rb") as f:
            content = f.read()
        doc_name = os.path.basename(path)
        doc_type = os.path.splitext(path)[1].lstrip(".").lower() or "bin"
    else:
        text = input("   Paste document text: ")
        if not text.strip():
            print("   [INFO] Nothing entered, cancelled.")
            return
        doc_name = input("   Name for this document: ").strip() or "untitled.txt"
        content = text
        doc_type = "text"

    doc_id = add_vendor_document(user_id, doc_name, content, doc_type=doc_type, notes=notes)
    print(f"   [OK] Saved '{doc_name}' as document #{doc_id}")

def main():
    print("=" * 60)
    print("[APP] Chatbot with SQLite Database")
    print("=" * 60)

    init_database()

    user_id, user_name, user_type = select_user()

    try:
        model, tokenizer, device = setup_model()
    except Exception as e:
        print(f"[WARN] Failed to initialize model: {e}")
        print("[INFO] Running in demo mode with pre-programmed responses...")
        model = None

    stats = get_user_stats(user_id)
    if stats['user_info']:
        print(f"\n[STATS] {stats['user_info'][0]} - {stats['chat_count']} past conversations")

    print("\n" + "=" * 60)
    banner = "[CHAT] Chat started! Type 'quit' to exit, 'history' to see chat history"
    if user_type == "vendor":
        banner += ", 'docs' to list your documents, 'adddoc' to upload one"
    print(banner)
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print("\nGoodbye!")
                break
            if user_input.lower() == 'history':
                history = get_chat_history(user_id, limit=10)
                if not history:
                    print("   No chat history yet.")
                else:
                    for i, (msg, resp, timestamp) in enumerate(reversed(history), 1):
                        print(f"{i}. [{timestamp}]\n   You: {msg}\n   Bot: {resp}\n")
                continue
            if user_type == "vendor" and user_input.lower() == 'docs':
                _handle_docs_command(user_id)
                continue
            if user_type == "vendor" and user_input.lower() == 'adddoc':
                _handle_adddoc_command(user_id)
                continue

            if model is not None:
                response = generate_response_with_tools(model, tokenizer, device, user_input, current_user_id=user_id, current_user_type=user_type)
            else:
                response = get_demo_response(user_input)
            print(f"Bot: {response}")

            save_chat_message(user_id, user_input, response)

        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")
            print("Please try again or type 'quit' to exit.")

    print("\n[SAVE] All conversations have been saved to 'chatbot.db'")

if __name__ == "__main__":
    main()