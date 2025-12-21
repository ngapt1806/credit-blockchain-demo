# =======================================================================
# credit_blockchain_demo.py
# HỆ THỐNG CHIA SẺ DỮ LIỆU TÍN DỤNG (Blockchain Chain + Streamlit)
# Ngân hàng A ghi sự kiện tín dụng | Ngân hàng B gửi yêu cầu & thẩm định | KH cấp/từ chối/thu hồi
# + SỔ CÁI CÔNG KHAI (Public Ledger)
#
# YÊU CẦU MỚI:
# - Mỗi giao dịch có request_id
# - Có chữ ký số (giả lập RSA)
# - Có hàm kiểm tra tính toàn vẹn chuỗi (hash + chữ ký)
# - Giao diện Sổ cái: chỉ tổng quan block + nút kiểm tra toàn vẹn
# =======================================================================

import time
import random
import datetime
import json
import hashlib
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="Hệ thống chia sẻ dữ liệu tín dụng", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
CHAIN_FILE = BASE_DIR / "chain.json"

# -----------------------------------------------------------------------
# TIMEZONE (VN)
# -----------------------------------------------------------------------
VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

# -----------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------
def generate_customer_id():
    return f"CUS_{random.randint(100000, 999999)}"

def generate_tx_hash():
    return "0x" + f"{random.getrandbits(128):032x}"

def generate_request_id():
    # request_id dùng chung cho tất cả loại giao dịch (TRANSACTION/REQUEST/CONSENT/LOG)
    return "REQ-" + f"{random.getrandbits(64):016x}"

def format_time(ts: int):
    """Hiển thị đúng giờ Việt Nam (UTC+7)."""
    try:
        ts = int(ts)
        return datetime.datetime.fromtimestamp(ts, tz=VN_TZ).strftime("%d/%m/%Y %H:%M:%S")
    except Exception:
        return "-"

def credit_decision(score: int):
    if score >= 750:
        return "🟢 Rất tốt", "ĐỀ XUẤT DUYỆT VAY – HẠN MỨC CAO", "success"
    elif score >= 700:
        return "🟢 Tốt", "ĐỀ XUẤT DUYỆT VAY", "success"
    elif score >= 650:
        return "🟡 Trung bình", "CÂN NHẮC – BỔ SUNG HỒ SƠ", "warning"
    else:
        return "🔴 Rủi ro cao", "TỪ CHỐI VAY", "error"

# -----------------------------------------------------------------------
# FAKE RSA SIGNATURE (SIMULATION) - KHÔNG DÙNG THƯ VIỆN NGOÀI
# - Tạo keypair RSA nhỏ (demo), ký bằng modular exponent
# - Dùng SHA-256(payload) -> int -> ký
# -----------------------------------------------------------------------
def _is_probable_prime(n: int, k: int = 8) -> bool:
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    # Miller-Rabin
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def _try(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    # random bases
    for _ in range(k):
        a = random.randrange(2, n - 2)
        if not _try(a):
            return False
    return True

def _gen_prime(bits: int = 128) -> int:
    while True:
        x = random.getrandbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(x):
            return x

def _egcd(a: int, b: int):
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y

def _modinv(a: int, m: int) -> int:
    g, x, _ = _egcd(a, m)
    if g != 1:
        raise ValueError("No modular inverse")
    return x % m

def rsa_generate_keypair(bits: int = 128):
    # demo: bits nhỏ cho chạy nhanh
    p = _gen_prime(bits)
    q = _gen_prime(bits)
    while q == p:
        q = _gen_prime(bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    if phi % e == 0:
        e = 3
    d = _modinv(e, phi)
    return {"n": n, "e": e, "d": d}

def _payload_to_int(payload: dict) -> int:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h, 16)

def rsa_sign(payload: dict, priv: dict) -> str:
    m = _payload_to_int(payload)
    sig = pow(m, int(priv["d"]), int(priv["n"]))
    return hex(sig)

def rsa_verify(payload: dict, signature_hex: str, pub: dict) -> bool:
    try:
        m = _payload_to_int(payload)
        sig = int(signature_hex, 16)
        check = pow(sig, int(pub["e"]), int(pub["n"]))
        return check == m
    except Exception:
        return False

# tạo “keypair hệ thống” cố định cho demo trong session
if "rsa_keys" not in st.session_state:
    kp = rsa_generate_keypair(bits=128)
    st.session_state.rsa_keys = {"n": kp["n"], "e": kp["e"], "d": kp["d"]}

RSA_KEYS = st.session_state.rsa_keys
RSA_PUB = {"n": RSA_KEYS["n"], "e": RSA_KEYS["e"]}  # public key

# -----------------------------------------------------------------------
# PUBLIC LEDGER HELPERS
# -----------------------------------------------------------------------
def _short_hash(s: str, head=10, tail=8) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= head + tail + 1:
        return s
    return s[:head] + "…" + s[-tail:]

def summarize_tx_public(tx: dict) -> str:
    """Tóm tắt giao dịch theo kiểu public (không lộ dữ liệu nhạy cảm)."""
    t = str(tx.get("type", "")).upper()
    if t == "SYSTEM":
        return "SYSTEM INIT"
    if t == "TRANSACTION":
        return f"TX: {tx.get('status_label', 'Giao dịch tín dụng')}"
    if t == "ACCESS_REQUEST":
        return "REQUEST: Yêu cầu truy cập"
    if t == "CONSENT":
        act = str(tx.get("action", "")).upper()
        return f"CONSENT: {act}"
    if t == "ACCESS_LOG":
        return "ACCESS LOG: Hồ sơ được truy cập"
    return f"{t}"

def build_public_ledger_df(bc) -> pd.DataFrame:
    rows = []
    for b in bc.chain:
        txs = b.transactions or []
        if not txs:
            content = "—"
        elif len(txs) == 1:
            content = summarize_tx_public(txs[0])
        else:
            content = f"{len(txs)} giao dịch (vd: {summarize_tx_public(txs[0])})"

        rows.append({
            "Block Index": b.index,
            "Thời gian": format_time(b.timestamp),
            "Nội dung giao dịch": content,
            "Hash ID": _short_hash(b.hash, 10, 8),
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------
# TRANSACTION SIGNING HELPERS
# -----------------------------------------------------------------------
SIGN_FIELDS_EXCLUDE = {"signature"}  # chữ ký không được tự ký vào chính nó

def tx_payload_for_sign(tx: dict) -> dict:
    # payload chuẩn để ký/verify (loại bỏ signature)
    return {k: tx[k] for k in sorted(tx.keys()) if k not in SIGN_FIELDS_EXCLUDE}

def sign_tx_inplace(tx: dict):
    payload = tx_payload_for_sign(tx)
    tx["signature"] = rsa_sign(payload, RSA_KEYS)

def verify_tx(tx: dict) -> bool:
    sig = tx.get("signature")
    if not sig:
        return False
    payload = tx_payload_for_sign(tx)
    return rsa_verify(payload, str(sig), RSA_PUB)

# -----------------------------------------------------------------------
# BLOCKCHAIN CORE
# -----------------------------------------------------------------------
class Block:
    def __init__(self, index, previous_hash, timestamp, transactions, nonce=0, block_hash=None):
        self.index = int(index)
        self.previous_hash = str(previous_hash)
        self.timestamp = int(timestamp)
        self.transactions = transactions
        self.nonce = int(nonce)
        self.hash = block_hash or self.calculate_hash()

    def calculate_hash(self) -> str:
        payload = json.dumps(
            {
                "index": self.index,
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp,
                "transactions": self.transactions,
                "nonce": self.nonce,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def mine(self, difficulty=2):
        target = "0" * int(difficulty)
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.calculate_hash()

    def to_dict(self):
        return {
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "nonce": self.nonce,
            "hash": self.hash,
        }

    @staticmethod
    def from_dict(d: dict):
        return Block(
            index=d["index"],
            previous_hash=d["previous_hash"],
            timestamp=d["timestamp"],
            transactions=d.get("transactions", []),
            nonce=d.get("nonce", 0),
            block_hash=d.get("hash"),
        )

class Blockchain:
    def __init__(self, difficulty=2):
        self.difficulty = int(difficulty)
        self.chain = [self._create_genesis_block()]
        self.pending = []
        self.access_rights = {}  # key: f"{customer_id}_{bank}" -> bool

    def _create_genesis_block(self):
        tx = {"type": "SYSTEM", "msg": "GENESIS", "time": int(time.time()), "request_id": "REQ-GENESIS"}
        sign_tx_inplace(tx)
        return Block(index=0, previous_hash="0", timestamp=int(time.time()), transactions=[tx])

    def add_transaction(self, tx: dict):
        tx = dict(tx)
        tx.setdefault("time", int(time.time()))
        tx.setdefault("request_id", generate_request_id())
        # ký số transaction
        sign_tx_inplace(tx)
        self.pending.append(tx)

    def mine_pending(self):
        if not self.pending:
            return None

        new_block = Block(
            index=len(self.chain),
            previous_hash=self.chain[-1].hash,
            timestamp=int(time.time()),
            transactions=self.pending,
        )
        new_block.mine(self.difficulty)
        self.chain.append(new_block)

        # update access rights from CONSENT tx
        for tx in self.pending:
            if tx.get("type") == "CONSENT":
                key = f"{tx.get('customer_id')}_{tx.get('target_bank')}"
                action = str(tx.get("action", "")).upper()
                self.access_rights[key] = (action == "GRANT")

        self.pending = []
        return new_block

    def rebuild_access_rights(self):
        self.access_rights = {}
        for b in self.chain:
            for tx in b.transactions:
                if tx.get("type") == "CONSENT":
                    key = f"{tx.get('customer_id')}_{tx.get('target_bank')}"
                    action = str(tx.get("action", "")).upper()
                    self.access_rights[key] = (action == "GRANT")

    def check_permission(self, customer_id: str, bank_name: str) -> bool:
        key = f"{customer_id}_{bank_name}"
        return bool(self.access_rights.get(key, False))

    def save(self, path=CHAIN_FILE):
        data = [b.to_dict() for b in self.chain]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path=CHAIN_FILE, difficulty=2):
        bc = Blockchain(difficulty=difficulty)
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                bc.chain = [Block.from_dict(x) for x in raw]
            except Exception:
                bc = Blockchain(difficulty=difficulty)
        bc.rebuild_access_rights()
        return bc

    # Helpers
    def iter_txs(self):
        for b in self.chain:
            for tx in b.transactions:
                yield (b, tx)

    def list_customers(self):
        s = set()
        for _, tx in self.iter_txs():
            cid = tx.get("customer_id")
            if cid:
                s.add(str(cid))
        return sorted(list(s))

    def customer_transactions(self, customer_id: str):
        rows = []
        for b, tx in self.iter_txs():
            if str(tx.get("customer_id", "")) == str(customer_id) and tx.get("type") == "TRANSACTION":
                rows.append((b, tx))
        rows.sort(key=lambda x: int(x[1].get("time", 0)))
        return rows

    def access_logs(self, customer_id: str):
        rows = []
        for b, tx in self.iter_txs():
            if str(tx.get("customer_id", "")) == str(customer_id) and tx.get("type") == "ACCESS_LOG":
                rows.append((b, tx))
        rows.sort(key=lambda x: int(x[1].get("time", 0)))
        return rows

    def latest_access_request(self, customer_id: str, requester_bank: str):
        """Request mới nhất + pending nếu sau request chưa có CONSENT."""
        latest_req = None
        for _, tx in self.iter_txs():
            if (
                tx.get("type") == "ACCESS_REQUEST"
                and str(tx.get("customer_id")) == str(customer_id)
                and str(tx.get("requester_bank")) == str(requester_bank)
            ):
                t = int(tx.get("time", 0))
                if latest_req is None or t > int(latest_req.get("time", 0)):
                    latest_req = dict(tx)

        if not latest_req:
            return None

        req_time = int(latest_req.get("time", 0))
        handled = False
        handled_action = None
        handled_time = None

        for _, tx in self.iter_txs():
            if (
                tx.get("type") == "CONSENT"
                and str(tx.get("customer_id")) == str(customer_id)
                and str(tx.get("target_bank")) == str(requester_bank)
            ):
                t = int(tx.get("time", 0))
                if t >= req_time:
                    handled = True
                    handled_action = str(tx.get("action", "")).upper()
                    handled_time = t

        latest_req["pending"] = not handled
        latest_req["handled_action"] = handled_action
        latest_req["handled_time"] = handled_time
        return latest_req

    def customer_loan_state(self, customer_id: str):
        txs = self.customer_transactions(customer_id)
        has_open = False
        last_event = None
        for _, tx in txs:
            s = int(tx.get("repayment_status", 0))
            last_event = tx
            if s == 0:
                has_open = True
            elif s in (1, 2):
                has_open = False
        return {"has_open": has_open, "last_event": last_event}

# -----------------------------------------------------------------------
# CHAIN INTEGRITY CHECK
# - check previous_hash linking
# - check block hash recalculation
# - check tx signature verify
# -----------------------------------------------------------------------
def verify_chain_integrity(bc: Blockchain):
    issues = []
    ok = True

    for i, b in enumerate(bc.chain):
        # 1) previous hash link
        if i == 0:
            if b.previous_hash != "0":
                ok = False
                issues.append(f"Block 0 previous_hash sai: {b.previous_hash}")
        else:
            prev = bc.chain[i - 1]
            if b.previous_hash != prev.hash:
                ok = False
                issues.append(f"Block {b.index} previous_hash không khớp (expected {prev.hash}, got {b.previous_hash})")

        # 2) block hash verify
        recalculated = b.calculate_hash()
        if b.hash != recalculated:
            ok = False
            issues.append(f"Block {b.index} hash không khớp (stored {b.hash}, recalculated {recalculated})")

        # 3) tx signature verify
        for j, tx in enumerate(b.transactions or []):
            if not verify_tx(tx):
                ok = False
                rid = tx.get("request_id", "N/A")
                t = tx.get("type", "N/A")
                issues.append(f"Chữ ký TX sai: Block {b.index}, tx#{j}, type={t}, request_id={rid}")

    return ok, issues

# -----------------------------------------------------------------------
# SCORING
# -----------------------------------------------------------------------
def calculate_onchain_score_from_chain(bc: Blockchain, customer_id: str):
    base = 650
    txs = bc.customer_transactions(customer_id)
    if not txs:
        return base, {"Đúng hạn": 0, "Trễ hạn": 0, "Đang vay": 0}

    ontime = late = 0
    for _, tx in txs:
        s = int(tx.get("repayment_status", 0))
        if s == 1:
            ontime += 1
        elif s == 2:
            late += 1

    state = bc.customer_loan_state(customer_id)
    open_flag = 1 if state["has_open"] else 0

    score = base + ontime * 50 - late * 50 + open_flag * 10
    score = max(300, min(850, score))
    return score, {"Đúng hạn": ontime, "Trễ hạn": late, "Đang vay": open_flag}

# -----------------------------------------------------------------------
# SMART CONTRACT MÔ PHỎNG (Python)
# -----------------------------------------------------------------------
class CreditSharingContractSim:
    BANK_B = "Ngân hàng B"
    BANK_A = "Ngân hàng A"

    def __init__(self, bc: Blockchain):
        self.bc = bc

    def bank_b_send_access_request(self, customer_id: str, purpose: str = "Thẩm định tín dụng"):
        tx = {
            "type": "ACCESS_REQUEST",
            "customer_id": str(customer_id),
            "requester_bank": self.BANK_B,
            "purpose": str(purpose),
            "tx_hash": generate_tx_hash(),
            # request_id tự thêm + ký trong add_transaction()
        }
        self.bc.add_transaction(tx)
        self.bc.mine_pending()
        return tx

    def grant_consent_to_bank_b(self, customer_id: str):
        self.bc.add_transaction({"type": "CONSENT", "customer_id": str(customer_id), "target_bank": self.BANK_B, "action": "GRANT"})
        self.bc.mine_pending()

    def deny_consent_to_bank_b(self, customer_id: str):
        self.bc.add_transaction({"type": "CONSENT", "customer_id": str(customer_id), "target_bank": self.BANK_B, "action": "DENY"})
        self.bc.mine_pending()

    def revoke_consent_from_bank_b(self, customer_id: str):
        self.bc.add_transaction({"type": "CONSENT", "customer_id": str(customer_id), "target_bank": self.BANK_B, "action": "REVOKE"})
        self.bc.mine_pending()

    def is_allowed(self, customer_id: str, bank_name: str) -> bool:
        return self.bc.check_permission(str(customer_id), str(bank_name))

    def log_access(self, customer_id: str, viewer_bank: str):
        self.bc.add_transaction({"type": "ACCESS_LOG", "customer_id": str(customer_id), "viewer": str(viewer_bank), "msg": "Viewed Profile"})
        self.bc.mine_pending()

    def record_transaction_bank_a(self, customer_id: str, amount: int, repayment_status: int, status_label: str):
        tx = {
            "type": "TRANSACTION",
            "bank": self.BANK_A,
            "customer_id": str(customer_id),
            "amount": int(amount),
            "repayment_status": int(repayment_status),
            "status_label": str(status_label),
            "tx_hash": generate_tx_hash(),
            # request_id tự thêm + ký trong add_transaction()
        }
        self.bc.add_transaction(tx)
        new_block = self.bc.mine_pending()
        return tx, new_block

    def bank_b_query_and_score(self, customer_id: str):
        cid = str(customer_id)
        if not self.is_allowed(cid, self.BANK_B):
            return None

        self.log_access(cid, self.BANK_B)

        score, detail = calculate_onchain_score_from_chain(self.bc, cid)
        rating, decision, level = credit_decision(int(score))
        return {
            "score": int(score),
            "detail": detail,
            "rating": rating,
            "decision": decision,
            "level": level,
            "tx_rows": self.bc.customer_transactions(cid),
        }

# -----------------------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------------------
if "bc" not in st.session_state:
    st.session_state.bc = Blockchain.load(difficulty=2)
if "new_customer_id" not in st.session_state:
    st.session_state.new_customer_id = generate_customer_id()
if "active_customer" not in st.session_state:
    st.session_state.active_customer = None

bc: Blockchain = st.session_state.bc
contract = CreditSharingContractSim(bc)

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🔗 Hệ thống chia sẻ dữ liệu tín dụng (Blockchain Chain)</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    menu = st.radio(
        "Chọn màn hình",
        [
            "1. Ngân hàng A - Ghi giao dịch",
            "2. Khách hàng (User App)",
            "3. Ngân hàng B - Gửi yêu cầu & Tra cứu",
            "4. Sổ cái (Public Ledger)",
        ],
    )

    if st.button("🧹 Reset demo", use_container_width=True):
        st.session_state.bc = Blockchain(difficulty=2)
        st.session_state.new_customer_id = generate_customer_id()
        st.session_state.active_customer = None
        try:
            if CHAIN_FILE.exists():
                CHAIN_FILE.unlink()
        except Exception:
            pass
        st.toast("Đã reset hệ thống", icon="✅")
        st.rerun()

bc = st.session_state.bc
contract = CreditSharingContractSim(bc)

# -----------------------------------------------------------------------
# 1) NGÂN HÀNG A: GHI SỰ KIỆN TÍN DỤNG
# -----------------------------------------------------------------------
if menu.startswith("1."):
    st.subheader("🏦 Ngân hàng A: Ghi nhận sự kiện tín dụng (On-chain)")

    col1, col2 = st.columns(2)

    with col1:
        mode = st.radio("Khách hàng", ["Tạo mới", "Chọn có sẵn"], horizontal=True)

        if mode == "Tạo mới":
            st.success(f"ID mới: {st.session_state.new_customer_id}")
            if st.button("🔄 Tạo ID khác"):
                st.session_state.new_customer_id = generate_customer_id()
                st.rerun()
            customer_id = st.session_state.new_customer_id
        else:
            customers = bc.list_customers()
            customers = [c for c in customers if len(bc.customer_transactions(c)) > 0]
            if not customers:
                st.warning("Chưa có khách hàng có giao dịch. Hãy tạo mới trước.")
                st.stop()
            customer_id = st.selectbox("Chọn ID", customers)

    with col2:
        amount = st.number_input("Số tiền (VND)", min_value=1_000_000, step=1_000_000)

        event_map = {
            "Giải ngân (mở khoản vay)": (0, "Giải ngân - mở khoản vay"),
            "Trả đúng hạn": (1, "Trả nợ đúng hạn"),
            "Trả trễ hạn": (2, "Trả nợ trễ hạn"),
        }
        event = st.selectbox("Loại sự kiện", list(event_map.keys()))

        if st.button("📤 Ghi giao dịch", use_container_width=False):
            cid = str(customer_id)
            repayment_status, status_label = event_map[event]

            cur_state = bc.customer_loan_state(cid)
            has_open = cur_state["has_open"]

            if repayment_status == 0 and has_open:
                st.warning("Lưu ý: Hệ thống đang coi khách có khoản vay 'đang mở'. Bạn vẫn có thể ghi 'Giải ngân' nếu đây là dữ liệu lịch sử/ngoại lệ.")
            if repayment_status in (1, 2) and (not has_open):
                st.warning("Lưu ý: Chưa thấy 'Giải ngân' trước đó. Bạn vẫn có thể ghi 'Trả đúng/trễ hạn' nếu đang nhập lịch sử.")

            tx, new_block = contract.record_transaction_bank_a(
                customer_id=cid,
                amount=int(amount),
                repayment_status=int(repayment_status),
                status_label=str(status_label),
            )
            bc.save()

            st.session_state.active_customer = cid
            if mode == "Tạo mới":
                st.session_state.new_customer_id = generate_customer_id()

            st.success("✅ Ghi nhận thành công")
            if new_block:
                # tx ở đây là bản gốc trước khi add_transaction() ký + request_id
                st.info("Giao dịch đã được đóng gói, ký số và ghi vào chuỗi.")
                st.code(f"TX Hash: {tx.get('tx_hash','-')}\nTime: {format_time(int(time.time()))}")

# -----------------------------------------------------------------------
# 2) KHÁCH HÀNG
# -----------------------------------------------------------------------
elif menu.startswith("2."):
    st.subheader("👤 Khách hàng: Nhận yêu cầu & quản lý quyền chia sẻ")

    customers = bc.list_customers()
    customers = [c for c in customers if len(bc.customer_transactions(c)) > 0]
    if not customers:
        st.info("Chưa có khách hàng. Hãy sang 'Ngân hàng A' để tạo giao dịch trước.")
        st.stop()

    default_idx = 0
    if st.session_state.active_customer in customers:
        default_idx = customers.index(st.session_state.active_customer)

    cid = st.selectbox("Chọn khách hàng", customers, index=default_idx)
    st.session_state.active_customer = str(cid)

    st.success(f"Khách hàng hiện tại: **{cid}**")

    # ✅ CHỈ HIỂN THỊ ĐIỂM
    score, _detail = calculate_onchain_score_from_chain(bc, cid)
    st.markdown("### 📈 Điểm tín dụng")
    st.metric("Điểm tín dụng", int(score))

    # ✅ Request từ NH B (chỉ hiện UI khi có request)
    req = bc.latest_access_request(cid, CreditSharingContractSim.BANK_B)

    if req:
        st.markdown("### 📨 Yêu cầu truy cập")

        if req.get("pending"):
            st.warning(
                f"**PENDING** | {format_time(req.get('time', 0))} | Mục đích: {req.get('purpose', '-')}"
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ CẤP QUYỀN", use_container_width=True):
                    contract.grant_consent_to_bank_b(cid)
                    bc.save()
                    st.toast("🔐 Đã cấp quyền", icon="✅")
                    st.rerun()
            with c2:
                if st.button("❌ TỪ CHỐI", use_container_width=True):
                    contract.deny_consent_to_bank_b(cid)
                    bc.save()
                    st.toast("🚫 Đã từ chối", icon="⛔")
                    st.rerun()
            with c3:
                if st.button("🧹 THU HỒI (REVOKE)", use_container_width=True):
                    contract.revoke_consent_from_bank_b(cid)
                    bc.save()
                    st.toast("🔒 Đã thu hồi", icon="⛔")
                    st.rerun()
        else:
            action = req.get("handled_action") or "-"
            ht = req.get("handled_time")
            st.info(
                f"Đã xử lý yêu cầu | Kết quả: **{action}** | Lúc: {format_time(ht) if ht else '-'}"
            )

    st.markdown("### 📄 Lịch sử giao dịch")
    tx_rows = bc.customer_transactions(cid)
    view = []
    for _, tx in tx_rows:
        view.append(
            {
                "Thời gian": format_time(tx.get("time", 0)),
                "Sự kiện": tx.get("status_label", ""),
                "Số tiền (VND)": int(tx.get("amount", 0)),
                "TX Hash": _short_hash(tx.get("tx_hash", ""), 10, 6),
                "Request ID": tx.get("request_id", ""),
                "Sig✓": "OK" if verify_tx(tx) else "FAIL",
            }
        )
    st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------
# 3) NGÂN HÀNG B
# -----------------------------------------------------------------------
elif menu.startswith("3."):
    st.subheader("🏦 Ngân hàng B: Gửi yêu cầu & tra cứu tín dụng")

    customers = bc.list_customers()
    customers = [c for c in customers if len(bc.customer_transactions(c)) > 0]
    if not customers:
        st.info("Chưa có khách hàng. Hãy sang 'Ngân hàng A' để tạo giao dịch trước.")
        st.stop()

    pick_cid = st.selectbox("Khách hàng cần tra cứu", customers)
    st.session_state.active_customer = str(pick_cid)

    req = bc.latest_access_request(pick_cid, CreditSharingContractSim.BANK_B)
    allowed = contract.is_allowed(pick_cid, CreditSharingContractSim.BANK_B)

    left, right = st.columns([2, 3], gap="large")

    with left:
        st.markdown("### 📨 Trạng thái yêu cầu")
        if not req:
            st.write("Chưa gửi yêu cầu.")
        else:
            if req.get("pending"):
                st.warning(f"Đã gửi - đang chờ KH xử lý | {format_time(req.get('time',0))}")
            else:
                st.info(f"KH đã xử lý: **{req.get('handled_action','-')}** | {format_time(req.get('handled_time') or 0)}")

        purpose = st.text_input("Mục đích truy cập", value="Thẩm định tín dụng")
        if st.button("📨 GỬI YÊU CẦU XEM HỒ SƠ", use_container_width=True):
            contract.bank_b_send_access_request(pick_cid, purpose=purpose)
            bc.save()
            st.toast("Đã gửi yêu cầu cho khách hàng", icon="📨")
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔐 Quyền hiện tại")
        st.write("✅ Được cấp quyền" if allowed else "⛔ Chưa được cấp quyền")

    with right:
        st.markdown("### 📊 Kết quả tra cứu")
        run = st.button("🔍 TRA CỨU ĐIỂM TÍN DỤNG", use_container_width=True)

        if not allowed:
            st.error("⛔ Chưa có quyền truy cập. Hãy gửi yêu cầu và chờ khách hàng cấp quyền.")
        else:
            if run:
                result = contract.bank_b_query_and_score(pick_cid)
                if result is None:
                    st.error("⛔ Không có quyền truy cập.")
                    st.stop()
                bc.save()

                score = result["score"]
                detail = result["detail"]
                rating = result["rating"]
                decision = result["decision"]
                level = result["level"]
                tx_rows = result["tx_rows"]

                st.markdown("#### 📄 Lịch sử tín dụng")
                view = []
                for _, tx in tx_rows:
                    txh = tx.get("tx_hash", "")
                    view.append(
                        {
                            "Thời gian": format_time(tx.get("time", 0)),
                            "Sự kiện": tx.get("status_label", ""),
                            "TX Hash": _short_hash(txh, 10, 6),
                            "Request ID": tx.get("request_id", ""),
                            "Sig✓": "OK" if verify_tx(tx) else "FAIL",
                        }
                    )
                st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)

                st.markdown("#### 📈 Điểm & đánh giá")
                st.metric("Điểm tín dụng", int(score))

                msg = f"**Xếp hạng:** {rating}\n\n**Khuyến nghị:** {decision}"
                if level == "success":
                    st.success(msg)
                elif level == "warning":
                    st.warning(msg)
                else:
                    st.error(msg)

# -----------------------------------------------------------------------
# 4) PUBLIC LEDGER (ONLY OVERVIEW + INTEGRITY BUTTON)
# -----------------------------------------------------------------------
elif menu.startswith("4."):
    st.subheader("📜 Sổ cái (Public Ledger)")

    df = build_public_ledger_df(bc)
    if df.empty:
        st.info("Chưa có dữ liệu sổ cái.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### ✅ Kiểm tra toàn vẹn chuỗi (Hash + Chữ ký)")

    if st.button("🔎 KIỂM TRA TOÀN VẸN", use_container_width=True):
        ok, issues = verify_chain_integrity(bc)
        if ok:
            st.success("✅ Chuỗi hợp lệ: previous_hash đúng, hash đúng, chữ ký TX đều hợp lệ.")
        else:
            st.error("❌ Phát hiện lỗi toàn vẹn!")
            for x in issues[:30]:
                st.write("- " + x)
            if len(issues) > 30:
                st.caption(f"(Còn {len(issues)-30} lỗi khác…)")

    with st.expander("🔑 Public Key (demo)"):
        st.code(f"n = {RSA_PUB['n']}\ne = {RSA_PUB['e']}")

    with st.expander("📂 Xem file chain.json (raw)"):
        if CHAIN_FILE.exists():
            st.code(CHAIN_FILE.read_text(encoding="utf-8"))
        else:
            st.write("Chưa có dữ liệu, file chain.json chưa được tạo.")
