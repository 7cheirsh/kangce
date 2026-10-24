import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import json
import os
import hashlib
import secrets
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go

# ===================== 页面基础配置 =====================
st.set_page_config(page_title="康测 - 关节活动度居家自测", layout="centered")

# ===================== 深蓝色主题 CSS =====================
DARK_BLUE_CSS = """
<style>
:root {
    --navy: #0b1f3a;
    --navy-2: #12325c;
    --navy-3: #1c4a86;
    --accent: #2f81f7;
}
.stApp { background: linear-gradient(160deg, #f4f7fb 0%, #e8eef7 100%); }
.hero {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 60%, var(--navy-3) 100%);
    color: #ffffff; padding: 28px 24px; border-radius: 16px; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(11, 31, 58, 0.25);
}
.hero h1 { color: #ffffff; font-size: 1.6rem; margin: 0 0 6px 0; }
.hero p { color: #cfe0f7; margin: 0; font-size: 0.92rem; }
.login-card {
    background: #ffffff; border-radius: 16px; padding: 26px 26px 10px 26px;
    box-shadow: 0 10px 30px rgba(11, 31, 58, 0.12); border-top: 5px solid var(--navy-2);
}
.stButton > button {
    background: linear-gradient(135deg, var(--navy-2), var(--navy-3));
    color: #ffffff; border: none; border-radius: 10px;
    padding: 0.5rem 1.2rem; font-weight: 600;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--navy-3), var(--accent)); color: #ffffff;
}
.stTextInput input:focus, .stTextInput input:focus-visible {
    border-color: var(--navy-3) !important;
    box-shadow: 0 0 0 2px rgba(28, 74, 134, 0.2) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%);
}
section[data-testid="stSidebar"] * { color: #e8f0fb !important; }
.stTabs [aria-selected="true"] {
    color: var(--navy-2) !important; border-bottom-color: var(--navy-2) !important;
}
</style>
"""
st.markdown(DARK_BLUE_CSS, unsafe_allow_html=True)

# ===================== 文件路径 =====================
USERS_FILE = "users.json"
PATIENTS_FILE = "patients.json"
HISTORY_FILE = "history.json"

# ===================== 用户存储 =====================
def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        salt = secrets.token_hex(16)
        users = {"doctor": {"salt": salt, "password_hash": _hash_password("123456", salt)}}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        return users
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def verify_user(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    u = users[username]
    return _hash_password(password, u["salt"]) == u["password_hash"]

def register_user(username: str, password: str) -> tuple:
    users = load_users()
    if username in users:
        return False, "用户名已存在"
    if len(password) < 6:
        return False, "密码至少 6 位"
    salt = secrets.token_hex(16)
    users[username] = {"salt": salt, "password_hash": _hash_password(password, salt)}
    save_users(users)
    return True, "注册成功，请登录"

# ===================== 患者存储 =====================
def load_patients() -> dict:
    if not os.path.exists(PATIENTS_FILE):
        return {}
    try:
        with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_patients(data: dict):
    with open(PATIENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_user_patients(username: str) -> list:
    return load_patients().get(username, [])

def add_patient(username: str, patient: dict) -> dict:
    data = load_patients()
    patients = data.get(username, [])
    patient["id"] = secrets.token_hex(6)
    patient["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patients.append(patient)
    data[username] = patients
    save_patients(data)
    return patient

def delete_patient(username: str, patient_id: str):
    data = load_patients()
    data[username] = [p for p in data.get(username, []) if p["id"] != patient_id]
    save_patients(data)

# ===================== 历史记录存储 =====================
def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_history(records: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def add_history(username: str, patient: dict, side: str, mode: str, angle: float, level: str):
    records = load_history()
    records.append({
        "user": username,
        "patient_id": patient["id"],
        "patient_name": patient["name"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "side": "右侧" if side == "right" else "左侧",
        "mode": mode,
        "angle": angle,
        "level": level,
    })
    save_history(records)

def patient_history(username: str, patient_id: str) -> list:
    return [
        r for r in load_history()
        if r.get("user") == username and r.get("patient_id") == patient_id
    ]

# ===================== 业务常量与图像处理 =====================
REF_MIN = 0
REF_MAX = 150
NORMAL_MIN = 120
MILD_MIN = 90

def auto_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    result = img_bgr.astype(np.float32)
    b, g, r = cv2.split(result)
    b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
    if b_mean < 1e-6 or g_mean < 1e-6 or r_mean < 1e-6:
        return img_bgr
    gray = (b_mean + g_mean + r_mean) / 3.0
    b = np.clip(b * (gray / b_mean), 0, 255)
    g = np.clip(g * (gray / g_mean), 0, 255)
    r = np.clip(r * (gray / r_mean), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

# ===================== 角度计算 =====================
def calculate_raw_angle(a, b, c):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return round(angle, 1)

def raw_to_flexion(raw_angle):
    return round(180.0 - raw_angle, 1)

def get_upper_limb_points(landmarks, img_w, img_h, side="right"):
    pts = {}
    if side == "right":
        pts["shoulder"] = [int(landmarks[12].x * img_w), int(landmarks[12].y * img_h)]
        pts["elbow"]    = [int(landmarks[14].x * img_w), int(landmarks[14].y * img_h)]
        pts["wrist"]    = [int(landmarks[16].x * img_w), int(landmarks[16].y * img_h)]
    else:
        pts["shoulder"] = [int(landmarks[11].x * img_w), int(landmarks[11].y * img_h)]
        pts["elbow"]    = [int(landmarks[13].x * img_w), int(landmarks[13].y * img_h)]
        pts["wrist"]    = [int(landmarks[15].x * img_w), int(landmarks[15].y * img_h)]
    return pts

def draw_single_arm(frame, pts):
    cv2.line(frame, pts["shoulder"], pts["elbow"], (128, 0, 255), 3)
    cv2.line(frame, pts["elbow"], pts["wrist"], (128, 0, 255), 3)
    for k in ("shoulder", "elbow", "wrist"):
        cv2.circle(frame, pts[k], 6, (255, 255, 255), -1)

def status_of(angle):
    if angle > REF_MAX:
        return f"🟡 超过参考上限{REF_MAX}°", (0, 255, 255), "warning", "超过上限"
    if angle >= NORMAL_MIN:
        return "✅ 屈曲角度正常", (0, 255, 0), "success", "正常"
    if angle >= MILD_MIN:
        return "⚠️ 轻度屈曲受限", (0, 165, 255), "warning", "轻度受限"
    return "❗ 明显屈曲受限", (0, 0, 255), "error", "明显受限"

# ===================== 视频处理器 =====================
class RealTimeProcessor(VideoProcessorBase):
    def __init__(self, selected_side):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.side = selected_side
        self.current_angle = 0.0
        self.max_angle = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = auto_white_balance(img)
        img_h, img_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            pts = get_upper_limb_points(results.pose_landmarks.landmark, img_w, img_h, self.side)
            
            # 算出几何内角（伸直≈180，弯曲≈小角度）
            raw_angle = calculate_raw_angle(pts["shoulder"], pts["elbow"], pts["wrist"])
            
            # 算出真正的屈肘角度（伸直≈0，弯曲≈大角度）
            self.current_angle = raw_to_flexion(raw_angle)
            
            if self.current_angle > self.max_angle:
                self.max_angle = self.current_angle

            draw_single_arm(img, pts)
            
            # 🔴 在肘部直接绘制“屈伸角”（小角度），用显眼的红色
            cv2.putText(img, f"{self.current_angle}°",
                        (pts["elbow"][0] + 15, pts["elbow"][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
            # 左上角实时数据
            cv2.putText(img, f"当前: {self.current_angle}°", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"最大值: {self.max_angle}°", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            text, color, _, _ = status_of(self.current_angle)
            cv2.putText(img, text, (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame.from_ndarray(img, format="bgr24")

    def __del__(self):
        try:
            self.pose.close()
        except Exception:
            pass

# ===================== 登录页 =====================
def login_page():
    st.markdown("""
    <div class="hero">
        <h1>康测 · 关节活动度居家自测系统</h1>
        <p>肘关节屈曲角度测量 · 居家康复初筛参考</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        auth_mode = st.radio("操作", ["登录", "注册"], horizontal=True, label_visibility="collapsed")
        username = st.text_input("用户名", key="auth_user")
        password = st.text_input("密码", type="password", key="auth_pwd")

        if auth_mode == "登录":
            if st.button("登 录", use_container_width=True):
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "👤 患者信息"
                    st.rerun()
                else:
                    st.error("用户名或密码错误")
            st.caption("初始账号：doctor　密码：123456")
        else:
            password2 = st.text_input("确认密码", type="password", key="auth_pwd2")
            if st.button("注 册", use_container_width=True):
                if password != password2:
                    st.error("两次输入的密码不一致")
                else:
                    ok, msg = register_user(username, password)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
        st.markdown('</div>', unsafe_allow_html=True)

# ===================== 患者信息页 =====================
def patient_page():
    st.markdown(f"""
    <div class="hero">
        <h1>患者基本信息</h1>
        <p>账号：{st.session_state.username}　|　信息用于关联测量记录</p>
    </div>
    """, unsafe_allow_html=True)

    patients = get_user_patients(st.session_state.username)

    if patients:
        st.subheader("选择患者")
        options = {f"{p['name']}（{p.get('gender','')} / {p.get('age','')}岁）": p for p in patients}
        choice = st.selectbox("当前测量的患者", list(options.keys()))
        selected = options[choice]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 进入测量", use_container_width=True):
                st.session_state.current_patient = selected
                st.session_state.page = "📐 测量"
                st.rerun()
        with col2:
            if st.button("🗑️ 删除该患者", use_container_width=True):
                delete_patient(st.session_state.username, selected["id"])
                if st.session_state.get("current_patient", {}).get("id") == selected["id"]:
                    st.session_state.pop("current_patient", None)
                st.success("已删除")
                st.rerun()
        st.markdown("---")

    st.subheader("新增患者" if patients else "请填写患者基本信息")
    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("姓名 *")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("性别", ["男", "女", "其他"])
        with c2:
            age = st.number_input("年龄", min_value=0, max_value=120, value=30, step=1)
        with c3:
            affected = st.selectbox("患侧", ["右侧", "左侧", "双侧"])
        diagnosis = st.text_input("诊断 / 备注（可选）", placeholder="如：右肘骨折术后 3 个月")
        submitted = st.form_submit_button("保存并进入测量", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("姓名不能为空")
        else:
            patient = add_patient(st.session_state.username, {
                "name": name.strip(),
                "gender": gender,
                "age": int(age),
                "affected": affected,
                "diagnosis": diagnosis.strip(),
            })
            st.session_state.current_patient = patient
            st.session_state.page = "📐 测量"
            st.success(f"已保存患者：{patient['name']}")
            st.rerun()

# ===================== 测量页（整合了操作引导） =====================
def photo_mode(selected_side):
    st.subheader("📷 拍照测量")
    photo = st.camera_input("Take a photo")
    if not photo:
        return

    img_np = np.array(Image.open(photo))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = auto_white_balance(img_bgr)
    h, w = img_bgr.shape[:2]

    img_rgb_wb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5
    ) as pose:
        res = pose.process(img_rgb_wb)

    if not res.pose_landmarks:
        st.error("未识别到人体关键点，请摆正姿势，上半身完整入镜")
        return

    pts = get_upper_limb_points(res.pose_landmarks.landmark, w, h, selected_side)
    raw_angle = calculate_raw_angle(pts["shoulder"], pts["elbow"], pts["wrist"])
    
    # 真正屈伸角
    angle = raw_to_flexion(raw_angle)

    draw_single_arm(img_bgr, pts)
    
    # 🔴 在肘部用红色小角度标注
    cv2.putText(img_bgr, f"{angle}°",
                (pts["elbow"][0] + 15, pts["elbow"][1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    img_show = cv2.cvtColor(cv2.flip(img_bgr, 1), cv2.COLOR_BGR2RGB)
    st.image(img_show, caption="识别结果（镜像预览）", use_container_width=True)

    st.subheader("📊 测量结果")
    st.write(f"屈曲角度：**{angle} °**（伸直位≈0°）")
    st.write(f"参考范围：[{REF_MIN} ~ {REF_MAX} °]")
    text, _, level, level_cn = status_of(angle)
    getattr(st, level)(f"{text}（{angle}°）")

    photo_key = hashlib.md5(photo.getvalue()).hexdigest()
    if st.session_state.get("last_photo_key") != photo_key:
        add_history(st.session_state.username, st.session_state.current_patient,
                    selected_side, "拍照", angle, level_cn)
        st.session_state.last_photo_key = photo_key
        st.caption("✅ 已自动保存本次测量到历史记录")


def video_mode(selected_side):
    st.subheader("🎥 实时动态测量")
    st.info("活动肘关节，画面左上角实时显示当前/最大屈曲角度与评估状态；本轮最大角度以画面「最大值」为准。")

    if (
        "processor" not in st.session_state
        or getattr(st.session_state.processor, "side", None) != selected_side
    ):
        st.session_state.processor = RealTimeProcessor(selected_side)
    processor = st.session_state.processor

    ctx = webrtc_streamer(
        key=f"elbow-real-time-{selected_side}",
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置本轮最大角度", use_container_width=True):
            processor.max_angle = 0.0
            st.success("最大角度已重置")
    with col2:
        if st.button("💾 保存本轮最大角度", use_container_width=True):
            if processor.max_angle <= 0:
                st.warning("当前最大角度为 0，尚未测到有效数据")
            else:
                _, _, _, level_cn = status_of(processor.max_angle)
                add_history(st.session_state.username, st.session_state.current_patient,
                            selected_side, "视频", processor.max_angle, level_cn)
                st.success(f"已保存：{processor.max_angle}°（{level_cn}）")

    if ctx.video_processor:
        st.write(f"本轮最大屈曲角度：{ctx.video_processor.max_angle} °")
        text, _, level, _ = status_of(ctx.video_processor.max_angle)
        getattr(st, level)(text)


def measure_page():
    patient = st.session_state.get("current_patient")
    if not patient:
        st.warning("尚未选择患者，请先填写患者基本信息")
        if st.button("前往患者信息"):
            st.session_state.page = "👤 患者信息"
            st.rerun()
        return

    st.info(f"👤 当前患者：**{patient['name']}**（{patient.get('gender','')} / {patient.get('age','')}岁 / 患侧：{patient.get('affected','')}）"
            + (f"　诊断：{patient['diagnosis']}" if patient.get('diagnosis') else ""))
    if st.button("切换患者"):
        st.session_state.page = "👤 患者信息"
        st.rerun()

    # 操作引导
    first_visit_key = f"guide_seen_{st.session_state.username}"
    show_guide = not st.session_state.get(first_visit_key, False)
    
    with st.expander("📖 操作引导：如何正确站立与测量（点击展开/收起）", expanded=show_guide):
        st.markdown("""
        **① 站立姿势**
        - **侧对摄像头站立**，身体与镜头垂直，双脚与肩同宽。
        - **上半身完整入镜**：头顶到腰部以下都要在画面内，手臂活动不能出画。
        - 摄像头高度**与胸口齐平**，距离约 **1.5～2 米**。
        - 待测手臂自然下垂作为起始位，测量时再缓慢弯曲。

        **② 光线与环境**
        - **面向光源站立，避免背光**（背光会变剪影），避免单侧强光。
        - 避免冷暖光混用，系统已内置**自动白平衡**抵消整体偏色，但不能修复背光。
        - 背景简洁，不要多人入镜，穿短袖避免遮挡肘部。

        **③ 测量动作**
        - **伸直位**：手臂自然下垂伸直，应接近 **0°**（3～5° 残余属正常）。
        - **弯曲**：缓慢屈肘到最大角度并保持 1～2 秒。
        - 视频模式结束后，记得点击下方的"保存本轮最大角度"。
        """)
        if show_guide:
            if st.button("我知道了，收起引导"):
                st.session_state[first_visit_key] = True
                st.rerun()

    side_option = st.selectbox("请选择待测肘关节", ["右侧肘关节", "左侧肘关节"])
    side_key = "right" if side_option == "右侧肘关节" else "left"

    tab1, tab2 = st.tabs(["📷 拍照模式", "🎥 实时视频模式"])
    with tab1:
        photo_mode(side_key)
    with tab2:
        video_mode(side_key)

# ===================== 趋势图 =====================
def render_trend_chart(df: pd.DataFrame):
    if df.empty:
        return

    chart_df = df.copy()
    chart_df["时间_dt"] = pd.to_datetime(chart_df["时间"])
    chart_df = chart_df.sort_values("时间_dt")

    fig = go.Figure()
    for side, color in [("右侧", "#1c4a86"), ("左侧", "#e07b00")]:
        sub = chart_df[chart_df["侧别"] == side]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["时间_dt"],
            y=sub["屈曲角度(°)"],
            mode="lines+markers",
            name=f"{side}肘",
            line=dict(color=color, width=2.5),
            marker=dict(size=8, line=dict(color="white", width=1)),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>角度：%{y:.1f}°<extra></extra>",
        ))

    fig.add_hline(y=120, line_dash="dash", line_color="#2e9e5b",
                  annotation_text="正常下限 120°", annotation_position="top left")
    fig.add_hline(y=150, line_dash="dot", line_color="#c9a400",
                  annotation_text="参考上限 150°", annotation_position="bottom left")

    fig.update_layout(
        title="康复趋势 · 屈曲角度变化",
        xaxis_title="测量时间",
        yaxis_title="屈曲角度 (°)",
        yaxis=dict(range=[0, 180]),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================== 历史记录页 =====================
def history_page():
    patient = st.session_state.get("current_patient")
    if not patient:
        st.warning("请先选择患者")
        if st.button("前往患者信息"):
            st.session_state.page = "👤 患者信息"
            st.rerun()
        return

    st.markdown(f"""
    <div class="hero">
        <h1>历史测量记录</h1>
        <p>患者：{patient['name']}　|　{patient.get('gender','')} / {patient.get('age','')}岁 / 患侧：{patient.get('affected','')}</p>
    </div>
    """, unsafe_allow_html=True)

    records = patient_history(st.session_state.username, patient["id"])
    if not records:
        st.info("暂无历史记录。完成一次拍照测量或保存视频测量后，记录会出现在这里。")
        return

    df = pd.DataFrame(records)
    df = df[["time", "side", "mode", "angle", "level"]].rename(columns={
        "time": "时间", "side": "侧别", "mode": "模式", "angle": "屈曲角度(°)", "level": "评估"
    })
    df = df.sort_values("时间", ascending=False).reset_index(drop=True)

    render_trend_chart(df)

    if len(df) >= 2:
        recent = df.sort_values("时间", ascending=False).head(5)
        older = df.sort_values("时间", ascending=False).iloc[5:10]
        if not older.empty:
            delta = recent["屈曲角度(°)"].mean() - older["屈曲角度(°)"].mean()
            if delta > 3:
                st.success(f"📈 近期平均角度较之前提升 {delta:.1f}°，康复趋势向好")
            elif delta < -3:
                st.warning(f"📉 近期平均角度较之前下降 {abs(delta):.1f}°，建议关注并咨询康复师")
            else:
                st.info(f"➡️ 近期角度变化平稳（{delta:+.1f}°）")

    c1, c2, c3 = st.columns(3)
    c1.metric("记录总数", len(df))
    c2.metric("最大角度", f"{df['屈曲角度(°)'].max():.1f}°")
    c3.metric("最近一次", df["时间"].iloc[0][5:16])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ 导出 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"history_{patient['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("⚠️ 危险操作"):
        if st.checkbox("我确认要清空该患者的全部历史记录"):
            if st.button("清空该患者历史记录", use_container_width=True):
                all_records = load_history()
                remain = [
                    r for r in all_records
                    if not (r.get("user") == st.session_state.username
                            and r.get("patient_id") == patient["id"])
                ]
                save_history(remain)
                st.success("已清空该患者历史记录")
                st.rerun()

# ===================== 主入口 =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "📐 测量"

if not st.session_state.logged_in:
    login_page()
    st.stop()

st.markdown(f"""
<div class="hero">
    <h1>康测 · 肘关节居家活动度自测</h1>
    <p>当前用户：{st.session_state.username}　|　伸直位≈0°，弯曲越多数值越大</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 账户")
    st.write(f"👤 {st.session_state.username}")
    if st.button("退出登录", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.pop("processor", None)
        st.session_state.pop("last_photo_key", None)
        st.session_state.pop("current_patient", None)
        st.session_state.page = "📐 测量"
        st.rerun()
    st.markdown("---")

    nav_options = ["📐 测量", "👤 患者信息", "📜 历史记录"]
    page = st.radio(
        "功能导航", nav_options,
        index=nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("---")
    st.markdown("> 本工具仅作居家康复初筛参考，不能替代医院专业评估。")

if page == "👤 患者信息":
    patient_page()
elif page == "📜 历史记录":
    history_page()
else:
    measure_page()