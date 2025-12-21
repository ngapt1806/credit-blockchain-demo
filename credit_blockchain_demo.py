# =======================================================================
# credit_blockchain_demo.py
# HỆ THỐNG CHIA SẺ DỮ LIỆU TÍN DỤNG (Blockchain + Streamlit) — BẢN NÂNG CẤP
# - Liên kết ACCESS_REQUEST <-> CONSENT bằng request_id
# - Chữ ký số RSA cho giao dịch (Bank A, Bank B, Customer)
# - Kiểm tra toàn vẹn chuỗi khi load (hash, previous_hash, verify chữ ký)
# - ACCESS_LOG giàu thông tin: request_id, purpose, result
# =======================================================================

import time
import random
import datetime
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.express as px

# Crypto libs
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="Hệ thống chia sẻ dữ liệu tín dụng", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
CHAIN_FILE = BASE_DIR / "chain.json"
KEYS_DIR = BASE_DIR / "keys"
CUSTOMER_KEYS_DIR = KEYS_DIR / "customers"
BANK_KEYS_DIR = KEYS_DIR / "banks"
KEYS_DIR.mkdir(exist_ok=True)
CUSTOMER_KEYS_DIR.mkdir(exist_ok=True)
BANK_KEYS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------
def generate_customer_id():
    return f"CUS_{random.randint(100000, 999999)}"

def generate_tx_hash():
    return "0x" + f"{random.getrandbits(128):032x}"

def format_time(ts: int):
    try:
        return datetime.datetime.fromtimestamp(int(ts)).strftime("%d/%m/%Y %H:%M:%S")
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
# KEY MANAGEMENT
# -----------------------------------------------------------------------
class KeyManager:
    @staticmethod
    def _bank_key_path(bank_name: str) -> Path:
        safe = bank_name.replace(" ", "_")
        return BANK_KEYS_DIR / f"{safe}.pem"

    @staticmethod
    def _customer_key_path(customer_id: str) -> Path:
        safe = str(customer_id).replace(" ", "_")
        return CUSTOMER_KEYS_DIR / f"{safe}.pem"

    @staticmethod
    def get_or_create_bank_key(bank_name: str):
        p = KeyManager._bank_key_path(bank_name)
        if p.exists():
            return KeyManager._load_private_key(p)
        # create
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        KeyManager._save_private_key(priv, p)
        return priv

    @staticmethod
    def get_or_create_customer_key(customer_id: str):
        p = KeyManager._customer_key_path(customer_id)
        if p.exists():
            return KeyManager._load_private_key(p)
        priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        KeyManager._save_private_key(priv, p)
        return priv

    @staticmethod
    def _save_private_key(private_key, path: Path):
        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        path.write_text(pem.decode("utf-8"), encoding="utf-8")

    @staticmethod
    def _load_private_key(path: Path):
        pem = path.read_bytes()
        return serialization.load_pem_private_key(pem, password=None)

    @staticmethod
    def serialize_public_key(private_key) -> str:
        pub = private_key.public_key()
        pem = pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem.decode("utf-8")

    @staticmethod
    def sign_tx(private_key, tx: dict, fields_to_sign: Optional[List[str]] = None) -> dict:
        # Exclude signature fields from payload
        payload_tx = dict(tx)
        payload_tx.pop("signature", None)
        payload_tx.pop("sender_pubkey", None)
        payload_tx.pop("sender", None)

        if fields_to_sign:
            payload = json.dumps({k: payload_tx.get(k) for k in fields_to_sign}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        else:
            payload = json.dumps(payload_tx, sort_keys=True, ensure_ascii=False).encode("utf-8")

        signature = private_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        tx["signature"] = signature.hex()
        tx["sender_pubkey"] = KeyManager.serialize_public_key(private_key)
        return tx

    @staticmethod
    def verify_tx_signature(tx: dict, fields_to_sign: Optional[List[str]] = None) -> bool:
        sig_hex = tx.get("signature")
        pub_pem = tx.get("sender_pubkey")
        if not sig_hex or not pub_pem:
            return False

        payload_tx = dict(tx)
        payload_tx.pop("signature", None)
        payload_tx.pop("sender_pubkey", None)
        payload_tx.pop("sender", None)

        if fields_to_sign:
            payload = json.dumps({k: payload_tx.get(k) for k in fields_to_sign}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        else:
            payload = json.dumps(payload_tx, sort_keys=True, ensure_ascii=False).encode("utf-8")

        try:
            pub_key = serialization.load_pem_public_key(pub_pem.encode("utf-8"))
            pub_key.verify(bytes.fromhex(sig_hex), payload, padding.PKCS1v15(), hashes.SHA256())
            return True
        except Exception:
            return False

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

    def _merkle_root(self) -> str:
        # Simple merkle root for audit
        hashes = [hashlib.sha256(json.dumps(tx, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                  for tx in self.transactions]
        if not hashes:
            return hashlib.sha256(b"").hexdigest()
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            new_level = []
            for i in range(0, len(hashes), 2):
                new_level.append(hashlib.sha256((hashes[i] + hashes[i+1]).encode("utf-8")).hexdigest())
            hashes = new_level
        return hashes[0]

    def calculate_hash(self) -> str:
        payload = json.dumps(
            {
                "index": self.index,
                "previous_hash": self.previous_hash,
                "timestamp": self.timestamp,
                "merkle_root": self._merkle_root(),
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
        self.chain: List[Block] = [self._create_genesis_block()]
        self.pending: List[dict] = []
        self.access_rights: Dict[str, bool] = {}  # key: f"{customer_id}_{bank}" -> bool
        self.request_consent_map: Dict[str, dict] = {}  # request_id -> latest consent tx

    def _create_genesis_block(self):
        return Block(
            index=0,
            previous_hash="0",
            timestamp=int(time.time()),
            transactions=[{"type": "SYSTEM", "msg": "GENESIS"}],
        )

    def add_transaction(self, tx: dict, require_signature=True):
        tx = dict(tx)
        tx.setdefault("time", int(time.time()))
        # Verify signature for non-SYSTEM transactions
        if require_signature and tx.get("type") != "SYSTEM":
            if not KeyManager.verify_tx_signature(tx):
                raise ValueError("Invalid transaction signature")
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

        # Update access rights and request-consent mapping from CONSENT tx
        for tx in self.pending:
            if tx.get("type") == "CONSENT":
                key = f"{tx.get('customer_id')}_{tx.get('target_bank')}"
                action = str(tx.get("action", "")).upper()
                self.access_rights[key] = (action == "GRANT")
                rid = tx.get("request_id")
                if rid:
                    self.request_consent_map[str(rid)] = tx

        self.pending = []
        return new_block

    def rebuild_access_rights(self):
        self.access_rights = {}
        self.request_consent_map = {}
        for b in self.chain:
            for tx in b.transactions:
                if tx.get("type") == "CONSENT":
                    key = f"{tx.get('customer_id')}_{tx.get('target_bank')}"
                    action = str(tx.get("action", "")).upper()
                    self.access_rights[key] = (action == "GRANT")
                    rid = tx.get("request_id")
                    if rid:
                        self.request_consent_map[str(rid)] = tx

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
                # Integrity check
                for i in range(1, len(bc.chain)):
                    prev = bc.chain[i-1]
                    cur = bc.chain[i]
                    # Verify block link and hash
                    if cur.previous_hash != prev.hash:
                        raise ValueError(f"Chain broken at block {i}: previous_hash mismatch")
                    if cur.hash != cur.calculate_hash():
                        raise ValueError(f"Chain broken at block {i}: hash mismatch")
                    # Verify transaction signatures
                    for tx in cur.transactions:
                        if tx.get("type") != "SYSTEM":
                            if not KeyManager.verify_tx_signature(tx):
                                raise ValueError(f"Invalid signature in block {i}")
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

    def latest_score_record(self, customer_id: str):
        latest = None
        for _, tx in self.iter_txs():
            if tx.get("type") == "SCORE" and str(tx.get("customer_id")) == str(customer_id):
                t = int(tx.get("time", 0))
                if latest is None or t > int(latest.get("time", 0)):
                    latest = tx
        return latest

    def latest_access_request(self, customer_id: str, requester_bank: str):
        """Trả về request mới nhất + trạng thái 'pending' nếu chưa có CONSENT cho request_id đó."""
        latest_req = None
        for _, tx in self.iter_txs():
            if tx.get("type") == "ACCESS_REQUEST" and str(tx.get("customer_id")) == str(customer_id) and str(tx.get("requester_bank")) == str(requester_bank):
                t = int(tx.get("time", 0))
                if latest_req is None or t > int(latest_req.get("time", 0)):
                    latest_req = dict(tx)

        if not latest_req:
            return None

        rid = latest_req.get("request_id")
        handled_tx = None
        for _, tx in self.iter_txs():
            if tx.get("type") == "CONSENT" and tx.get("request_id") == rid:
                handled_tx = tx

        latest_req["pending"] = handled_tx is None
        latest_req["handled_action"] = handled_tx.get("action").upper() if handled_tx else None
        latest_req["handled_time"] = handled_tx.get("time") if handled_tx else None
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
    return score, {"Đúng hạn": ountime if 'ountime' in locals() else ontime, "Trễ hạn": late, "Đang vay": open_flag}

# -----------------------------------------------------------------------
# SMART CONTRACT SIMULATION
# -----------------------------------------------------------------------
class CreditSharingContractSim:
    BANK_B = "Ngân hàng B"
    BANK_A = "Ngân hàng A"

    def __init__(self, bc: Blockchain):
        self.bc = bc
        self.bank_a_key = KeyManager.get_or_create_bank_key(self.BANK_A)
        self.bank_b_key = KeyManager.get_or_create_bank_key(self.BANK_B)

    # --- Bank B gửi yêu cầu ---
    def bank_b_send_access_request(self, customer_id: str, purpose: str = "Thẩm định tín dụng"):
        request_id = generate_tx_hash()
        tx = {
            "type": "ACCESS_REQUEST",
            "request_id": request_id,
            "customer_id": str(customer_id),
            "requester_bank": self.BANK_B,
            "purpose": str(purpose),
            "tx_hash": generate_tx_hash(),
            "time": int(time.time()),
            "sender": self.BANK_B,
        }
        tx = KeyManager.sign_tx(self.bank_b_key, tx)
        self.bc.add_transaction(tx)
        self.bc.mine_pending()
        return tx

    # --- KH xử lý yêu cầu: cấp / từ chối / thu hồi ---
    def grant_consent_to_bank_b(self, customer_id: str, request_id: str):
        priv = KeyManager.get_or_create_customer_key(customer_id)
        tx = {
            "type": "CONSENT",
            "request_id": str(request_id),
            "customer_id": str(customer_id),
            "target_bank": self.BANK_B,
            "action": "GRANT",
            "time": int(time.time()),
            "sender": str(customer_id),
        }
        tx = KeyManager.sign_tx(priv, tx)
        self.bc.add_transaction(tx)
        self.bc.mine_pending()

    def deny_consent_to_bank_b(self, customer_id: str, request_id: str):
        priv = KeyManager.get_or_create_customer_key(customer_id)
        tx = {
            "type": "CONSENT",
            "request_id": str(request_id),
            "customer_id": str(customer_id),
            "target_bank": self.BANK_B,
            "action": "DENY",
            "time": int(time.time()),
            "sender": str(customer_id),
        }
        tx = KeyManager.sign_tx(priv, tx)
        self.bc.add_transaction(tx)
        self.bc.mine_pending()

    def revoke_consent_from_bank_b(self, customer_id: str):
        priv = KeyManager.get_or_create_customer_key(customer_id)
        tx = {
            "type": "CONSENT",
            "request_id": generate_tx_hash(),  # standalone revoke event
            "customer_id": str(customer_id),
            "target_bank": self.BANK_B,
            "action": "REVOKE",
            "time": int(time.time()),
            "sender": str(customer_id),
        }
        tx = KeyManager.sign_tx(priv, tx)
        self.bc.add_transaction(tx)
        self.bc.mine_pending()

    def is_allowed(self, customer_id: str, bank_name: str) -> bool:
        return self.bc.check_permission(str(customer_id), str(bank_name))

    def log_access(self, customer_id: str, viewer_bank: str, request_id: str, purpose: str, result: str = "SUCCESS"):
        tx = {
            "type": "ACCESS_LOG",
            "customer_id": str(customer_id),
            "viewer": str(viewer_bank),
            "request_id": str(request_id),
            "purpose": str(purpose),
            "result": str(result),
            "msg": "Viewed Profile",
            "time": int(time.time()),
            "sender": viewer_bank,
        }
        # Signed by the viewer bank
        key = self.bank_b_key if viewer_bank == self.BANK_B else self.bank_a_key
        tx = KeyManager.sign_tx(key, tx)
        self.bc.add_transaction(tx)
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
            "time": int(time.time()),
            "sender": self.BANK_A,
        }
        tx = KeyManager.sign_tx(self.bank_a_key, tx)
        self.bc.add_transaction(tx)
        new_block = self.bc.mine_pending()
        return tx, new_block

        # NOTE: no return after this line

    def write_score_record(self, customer_id: str, viewer_bank: str, score: int, detail: dict, rating: str, decision: str):
        key = self.bank_b_key if viewer_bank == self.BANK_B else self.bank_a_key
        tx = {
            "type": "SCORE",
            "customer_id": str(customer_id),
            "viewer_bank": str(viewer_bank),
            "score": int(score),
            "detail": dict(detail),
            "rating": str(rating),
            "decision": str(decision),
            "tx_hash": generate_tx_hash(),
            "time": int(time.time()),
            "sender": viewer_bank,
        }
        tx = KeyManager.sign_tx(key, tx)
        self.bc.add_transaction(tx)
        self.bc.mine_pending()
        return tx

    def bank_b_query_and_score(self, customer_id: str, purpose: str = "Thẩm định tín dụng"):
        cid = str(customer_id)
        if not self.is_allowed(cid, self.BANK_B):
            return None

        # Use the latest ACCESS_REQUEST for this customer to tie logs
        req = self.bc.latest_access_request(cid, self.BANK_B)
        request_id = req["request_id"] if req else generate_tx_hash()

        self.log_access(cid, self.BANK_B, request_id=request_id, purpose=purpose, result="SUCCESS")

        score, detail = calculate_onchain_score_from_chain(self.bc, cid)
        rating, decision, level = credit_decision(int(score))

        self.write_score_record(cid, self.BANK_B, int(score), detail, rating, decision)

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
    "<h1 style='text-align:center;'>🔗 Hệ thống chia sẻ dữ liệu tín dụng (Blockchain)</h1>",
    unsafe_allow_html=True,
)

with st.sidebar:
    menu = st.radio(
        "Chọn màn hình",
        [
            "1. Ngân hàng A - Ghi giao dịch",
            "2. Khách hàng (User App)",
            "3. Ngân hàng B - Gửi yêu cầu & Thẩm định",
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

            # Cảnh báo logic trạng thái
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
                st.code(f"TX Hash: {tx['tx_hash']}\nTime: {format_time(tx['time'])}")

# -----------------------------------------------------------------------
# 2) KHÁCH HÀNG: NHẬN YÊU CẦU & CẤP/TỪ CHỐI/THU HỒI + XEM ĐIỂM
# -----------------------------------------------------------------------
elif menu.startswith("2."):
    st.subheader("👤 Khách hàng: Nhận yêu cầu & quản lý quyền chia sẻ")

    customers = bc.list_customers()
    customers = [c for c in customers if len(bc.customer_transactions(c)) > 0]
    if not customers:
        st.info("Chưa có khách hàng. Hãy sang 'Ngân hàng A' để tạo giao dịch trước.")
        st.stop()

    # chọn KH (ưu tiên active)
    default_idx = 0
    if st.session_state.active_customer in customers:
        default_idx = customers.index(st.session_state.active_customer)

    cid = st.selectbox("Chọn khách hàng", customers, index=default_idx)
    st.session_state.active_customer = str(cid)

    st.success(f"Khách hàng hiện tại: **{cid}**")

    # đảm bảo có khóa của khách hàng
    _ = KeyManager.get_or_create_customer_key(cid)

    # trạng thái quyền hiện tại
    allowed = contract.is_allowed(cid, CreditSharingContractSim.BANK_B)
    st.info(f"Trạng thái hiện tại với Ngân hàng B: **{'ĐÃ CẤP QUYỀN' if allowed else 'CHƯA CẤP / ĐÃ TỪ CHỐI / ĐÃ THU HỒI'}**")

    # hiển thị yêu cầu mới nhất từ Bank B
    req = bc.latest_access_request(cid, CreditSharingContractSim.BANK_B)

    st.markdown("### 📨 Yêu cầu truy cập từ Ngân hàng B")
    if not req:
        st.write("— Chưa có yêu cầu nào từ Ngân hàng B.")
    else:
        rid = req.get("request_id")
        if req.get("pending"):
            st.warning(
                f"**PENDING** | Thời gian: {format_time(req.get('time',0))} | Mục đích: {req.get('purpose','-')} | Request ID: {rid}"
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("✅ CẤP QUYỀN", use_container_width=True):
                    contract.grant_consent_to_bank_b(cid, request_id=rid)
                    bc.save()
                    st.toast("🔐 Đã cấp quyền cho Ngân hàng B", icon="✅")
                    st.rerun()
            with c2:
                if st.button("❌ TỪ CHỐI", use_container_width=True):
                    contract.deny_consent_to_bank_b(cid, request_id=rid)
                    bc.save()
                    st.toast("🚫 Đã từ chối yêu cầu", icon="⛔")
                    st.rerun()
            with c3:
                if st.button("🧹 THU HỒI (nếu đã cấp)", use_container_width=True):
                    contract.revoke_consent_from_bank_b(cid)
                    bc.save()
                    st.toast("🔒 Đã thu hồi quyền", icon="⛔")
                    st.rerun()
        else:
            action = req.get("handled_action") or "-"
            ht = req.get("handled_time")
            st.info(
                f"Đã xử lý yêu cầu | Kết quả: **{action}** | Lúc: {format_time(ht) if ht else '-'} | Request ID: {rid}"
            )
            if st.button("🧹 THU HỒI QUYỀN (REVOKE)"):
                contract.revoke_consent_from_bank_b(cid)
                bc.save()
                st.toast("🔒 Đã thu hồi quyền", icon="⛔")
                st.rerun()

    # Điểm tín dụng mới nhất (on-chain SCORE)
    st.markdown("### 📈 Điểm tín dụng (mới nhất)")
    score_tx = bc.latest_score_record(cid)
    if not score_tx:
        st.write("— Chưa có điểm. (Ngân hàng B cần thẩm định để ghi điểm lên hệ thống.)")
    else:
        st.metric("Điểm tín dụng", int(score_tx.get("score", 0)))
        st.caption(f"Cập nhật: {format_time(score_tx.get('time',0))} | Bởi: {score_tx.get('viewer_bank','-')}")
        # biểu đồ chi tiết nếu có
        detail = score_tx.get("detail", {})
        if isinstance(detail, dict) and len(detail) > 0:
            pie = pd.DataFrame(detail.items(), columns=["Loại", "Số lượng"])
            fig = px.pie(pie, values="Số lượng", names="Loại", hole=0.45)
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        # khuyến nghị
        rating = score_tx.get("rating", "")
        decision = score_tx.get("decision", "")
        st.info(f"**Xếp hạng:** {rating}\n\n**Khuyến nghị:** {decision}")

    # xem lịch sử giao dịch
    st.markdown("### 📄 Lịch sử giao dịch")
    tx_rows = bc.customer_transactions(cid)
    view = []
    for _, tx in tx_rows:
        view.append(
            {
                "Thời gian": format_time(tx.get("time", 0)),
                "Sự kiện": tx.get("status_label", ""),
                "Số tiền (VND)": int(tx.get("amount", 0)),
                "TX Hash": tx.get("tx_hash", ""),
            }
        )
    st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)

    with st.expander("🕵️ Nhật ký truy cập (Access Logs)"):
        logs = bc.access_logs(cid)
        if not logs:
            st.write("—")
        else:
            rows = []
            for _, tx in logs:
                rows.append({
                    "Thời gian": format_time(tx.get("time", 0)),
                    "Người xem": tx.get("viewer", ""),
                    "Mục đích": tx.get("purpose", ""),
                    "Request ID": tx.get("request_id", ""),
                    "Kết quả": tx.get("result", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------
# 3) NGÂN HÀNG B: GỬI YÊU CẦU -> CHỜ KH -> THẨM ĐỊNH (NẾU ĐƯỢC CẤP)
# -----------------------------------------------------------------------
elif menu.startswith("3."):
    st.subheader("🏦 Ngân hàng B: Gửi yêu cầu truy cập & thẩm định")

    customers = bc.list_customers()
    customers = [c for c in customers if len(bc.customer_transactions(c)) > 0]
    if not customers:
        st.info("Chưa có khách hàng. Hãy sang 'Ngân hàng A' để tạo giao dịch trước.")
        st.stop()

    pick_cid = st.selectbox("Chọn khách hàng cần thẩm định", customers)
    st.session_state.active_customer = str(pick_cid)

    # banner
    st.markdown(
        f"""
        <div style="
            background:#e9f7ef;
            border:1px solid #cdeccd;
            padding:14px 18px;
            border-radius:12px;
            color:#1b5e20;
            font-size:18px;
            font-weight:600;
            width:100%;
            margin: 0 0 14px 0;
        ">
            Khách hàng: <span style="font-weight:800;">{pick_cid}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    req = bc.latest_access_request(pick_cid, CreditSharingContractSim.BANK_B)
    allowed = contract.is_allowed(pick_cid, CreditSharingContractSim.BANK_B)

    c1, c2 = st.columns([2, 3], gap="large")

    with c1:
        st.markdown("### 📨 Trạng thái yêu cầu")
        if not req:
            st.write("Chưa gửi yêu cầu.")
        else:
            if req.get("pending"):
                st.warning(f"Đã gửi - đang chờ KH xử lý | {format_time(req.get('time',0))} | Request ID: {req.get('request_id')}")
            else:
                st.info(f"Đã được KH xử lý: **{req.get('handled_action','-')}** | {format_time(req.get('handled_time') or 0)} | Request ID: {req.get('request_id')}")

        purpose = st.text_input("Mục đích truy cập", value="Thẩm định tín dụng")
        if st.button("📨 GỬI YÊU CẦU XEM HỒ SƠ", use_container_width=True):
            contract.bank_b_send_access_request(pick_cid, purpose=purpose)
            bc.save()
            st.toast("Đã gửi yêu cầu cho khách hàng", icon="📨")
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔐 Quyền hiện tại")
        st.write("✅ Được cấp quyền" if allowed else "⛔ Chưa được cấp quyền")

    with c2:
        st.markdown("### 🔍 Thẩm định & tính điểm")
        if not allowed:
            st.error("⛔ Chưa có quyền truy cập. Hãy gửi yêu cầu và chờ khách hàng cấp quyền.")
        else:
            if st.button("🔍 TRUY VẤN DỮ LIỆU & TÍNH ĐIỂM", use_container_width=True):
                purpose_val = purpose if isinstance(purpose, str) and purpose.strip() else "Thẩm định tín dụng"
                result = contract.bank_b_query_and_score(pick_cid, purpose=purpose_val)
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

                left, right = st.columns([3, 2], gap="large")

                with left:
                    st.markdown("#### 📄 Lịch sử tín dụng")
                    view = []
                    for _, tx in tx_rows:
                        txh = tx.get("tx_hash", "")
                        txh_short = (txh[:10] + "…" + txh[-6:]) if isinstance(txh, str) and len(txh) > 20 else txh
                        view.append(
                            {
                                "Thời gian": format_time(tx.get("time", 0)),
                                "Sự kiện": tx.get("status_label", ""),
                                "Số tiền (VND)": int(tx.get("amount", 0)),
                                "TX Hash": txh_short,
                            }
                        )
                    st.dataframe(pd.DataFrame(view), use_container_width=True, hide_index=True)

                with right:
                    st.markdown("#### 📈 Điểm & đánh giá")
                    st.metric("Điểm tín dụng", int(score))

                    pie = pd.DataFrame(detail.items(), columns=["Loại", "Số lượng"])
                    fig = px.pie(pie, values="Số lượng", names="Loại", hole=0.45)
                    fig.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    msg = f"**Xếp hạng:** {rating}\n\n**Khuyến nghị:** {decision}"
                    if level == "success":
                        st.success(msg)
                    elif level == "warning":
                        st.warning(msg)
                    else:
                        st.error(msg)

                st.toast("Đã ghi điểm lên hệ thống để KH xem ở mục 'Khách hàng (User App)'", icon="✅")
