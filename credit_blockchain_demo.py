# =======================================================================
# credit_blockchain_demo.py
# HỆ THỐNG CHIA SẺ DỮ LIỆU TÍN DỤNG (Blockchain + Streamlit)
# - Vai trò: Ngân hàng A (ghi giao dịch), Khách hàng (xử lý yêu cầu),
#   Ngân hàng B (yêu cầu & thẩm định)
# - Sổ cái (Public Ledger): hiển thị chuỗi block theo luồng 3 vai
# - YÊU CẦU SỬA:
#   1) Khách hàng: chỉ hiện Điểm tín dụng + Khuyến nghị (không biểu đồ).
#   2) Ngân hàng B: khi thẩm định, không hiện số tiền, chỉ hiện sự kiện + hash.
#   3) Public Ledger: chỉ bảng tổng quan block, không có chi tiết block.
# =======================================================================

import time
import random
import datetime
import json
import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px  # giữ import để bạn không bị lỗi nếu còn dùng nơi khác

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
st.set_page_config(page_title="Hệ thống chia sẻ dữ liệu tín dụng", layout="wide")
BASE_DIR = Path(__file__).resolve().parent
CHAIN_FILE = BASE_DIR / "chain.json"

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

def _short_hash(s: str, head=12, tail=8) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= head + tail + 3:
        return s
    return s[:head] + "..." + s[-tail:]

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
        return Block(
            index=0,
            previous_hash="0",
            timestamp=int(time.time()),
            transactions=[{"type": "SYSTEM", "msg": "GENESIS"}],
        )

    def add_transaction(self, tx: dict):
        tx = dict(tx)
        tx.setdefault("time", int(time.time()))
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
        """Trả về request mới nhất + trạng thái 'pending' nếu sau request chưa có CONSENT."""
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
# SMART CONTRACT SIMULATION
# -----------------------------------------------------------------------
class CreditSharingContractSim:
    BANK_B = "Ngân hàng B"
    BANK_A = "Ngân hàng A"

    def __init__(self, bc: Blockchain):
        self.bc = bc

    # --- Bank B gửi yêu cầu ---
    def bank_b_send_access_request(self, customer_id: str, purpose: str = "Thẩm định tín dụng"):
        tx = {
            "type": "ACCESS_REQUEST",
            "customer_id": str(customer_id),
            "requester_bank": self.BANK_B,
            "purpose": str(purpose),
            "tx_hash": generate_tx_hash(),
            "time": int(time.time()),
        }
        self.bc.add_transaction(tx)
        self.bc.mine_pending()
        return tx

    # --- KH xử lý yêu cầu: cấp / từ chối / thu hồi ---
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
            "amount": int(amount),  # vẫn ghi on-chain để demo đơn giản
            "repayment_status": int(repayment_status),
            "status_label": str(status_label),
            "tx_hash": generate_tx_hash(),
            "time": int(time.time()),
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
            "3. Ngân hàng B - Gửi yêu cầu & Thẩm định",
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
                st.warning("Lưu ý: Khách hiện có khoản vay 'đang mở'. Bạn vẫn có thể ghi 'Giải ngân' nếu đây là dữ liệu lịch sử/ngoại lệ.")
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
# 2) KHÁCH HÀNG: CHỈ HIỆN ĐIỂM + KHUYẾN NGHỊ (KHÔNG BIỂU ĐỒ)
# -----------------------------------------------------------------------
elif menu.startswith("2."):
    st.subheader("👤 Khách hàng: Xem điểm tín dụng")

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

    # Chỉ hiện yêu cầu truy cập nếu có yêu cầu
    req = bc.latest_access_request(cid, CreditSharingContractSim.BANK_B)
    if req and req.get("pending"):
        st.markdown("### 📨 Yêu cầu truy cập (ẩn danh)")
        st.warning(f"Thời gian: {format_time(req.get('time',0))} | Mục đích: {req.get('purpose','-')}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("✅ CẤP QUYỀN", use_container_width=True):
                contract.grant_consent_to_bank_b(cid)
                bc.save()
                st.toast("✅ Đã xử lý", icon="✅")
                st.rerun()
        with c2:
            if st.button("❌ TỪ CHỐI", use_container_width=True):
                contract.deny_consent_to_bank_b(cid)
                bc.save()
                st.toast("✅ Đã xử lý", icon="✅")
                st.rerun()
        with c3:
            if st.button("🧹 THU HỒI", use_container_width=True):
                contract.revoke_consent_from_bank_b(cid)
                bc.save()
                st.toast("✅ Đã xử lý", icon="✅")
                st.rerun()

    # ✅ Chỉ hiện điểm + khuyến nghị (không biểu đồ)
    st.markdown("### 📈 Điểm tín dụng")
    score, _detail = calculate_onchain_score_from_chain(bc, cid)
    rating, decision, level = credit_decision(int(score))

    st.metric("Điểm tín dụng", int(score))

    msg = f"**Xếp hạng:** {rating}\n\n**Khuyến nghị:** {decision}"
    if level == "success":
        st.success(msg)
    elif level == "warning":
        st.warning(msg)
    else:
        st.error(msg)

# -----------------------------------------------------------------------
# 3) NGÂN HÀNG B: THẨM ĐỊNH (KHÔNG HIỆN SỐ TIỀN)
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
                st.info(f"Đã được KH xử lý: **{req.get('handled_action','-')}** | {format_time(req.get('handled_time') or 0)}")

        purpose = st.text_input("Mục đích truy cập", value="Thẩm định tín dụng")
        if st.button("📨 GỬI YÊU CẦU XEM HỒ SƠ", use_container_width=True):
            contract.bank_b_send_access_request(pick_cid, purpose=purpose)
            bc.save()
            st.toast("Đã gửi yêu cầu", icon="📨")
            st.rerun()

        st.markdown("---")
        st.markdown("### 🔐 Quyền hiện tại")
        st.write("✅ Được cấp quyền" if allowed else "⛔ Chưa được cấp quyền")

    with right:
        st.markdown("### 🔍 Thẩm định & tính điểm")
        if not allowed:
            st.error("⛔ Chưa có quyền truy cập.")
        else:
            if st.button("🔍 TRUY VẤN & TÍNH ĐIỂM", use_container_width=True):
                result = contract.bank_b_query_and_score(pick_cid)
                if result is None:
                    st.error("⛔ Không có quyền truy cập.")
                    st.stop()
                bc.save()

                score = result["score"]
                rating = result["rating"]
                decision = result["decision"]
                level = result["level"]
                tx_rows = result["tx_rows"]

                st.markdown("#### 📄 Lịch sử tín dụng (không hiện số tiền)")
                view = []
                for _, tx in tx_rows:
                    txh = tx.get("tx_hash", "")
                    txh_short = _short_hash(txh, 10, 6)
                    view.append(
                        {
                            "Thời gian": format_time(tx.get("time", 0)),
                            "Sự kiện": tx.get("status_label", ""),
                            "TX Hash": txh_short,
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
# 4) SỔ CÁI (PUBLIC LEDGER): CHỈ BẢNG TỔNG QUAN BLOCK
# -----------------------------------------------------------------------
elif menu.startswith("4."):
    st.subheader("📒 Sổ cái (Public Ledger)")

    rows = []
    for block in bc.chain:
        if not block.transactions:
            content = "—"
        else:
            # tóm tắt nội dung theo type giao dịch đầu tiên trong block
            tx = block.transactions[0]
            t = str(tx.get("type", "")).upper()
            if t == "SYSTEM":
                content = "SYSTEM INIT"
            elif t == "TRANSACTION":
                content = f"TX: {tx.get('status_label', '')}"
            elif t == "ACCESS_REQUEST":
                content = "REQUEST: yêu cầu truy cập"
            elif t == "CONSENT":
                content = f"CONSENT: {str(tx.get('action','')).upper()}"
            elif t == "ACCESS_LOG":
                content = "ACCESS LOG: hồ sơ được truy cập"
            else:
                content = t

        rows.append({
            "Block": int(block.index),
            "Thời gian": format_time(block.timestamp),
            "Nội dung giao dịch": content,
            "Hash ID": _short_hash(block.hash, 12, 8),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
