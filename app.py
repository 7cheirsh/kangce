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
st.set_page_config(page_title="康测 - 居家康复与运动监测系统", layout="centered")

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

# ===================== 全局 WebRTC 配置 =====================
# 增加 TCP 穿透，确保手机或跨网络下能连上
RTC_CONFIG = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": "turn:openrelay.metered.ca:443?transport=tcp",
            "username": "openrelayproject",
            "credential": "openrelayproject",
        }
    ]
}

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

# ===================== 用户资料存储 =====================
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

def get_user_profiles(username: str) -> list:
    return load_patients().get(username, [])

def add_profile(username: str, profile: dict) -> dict:
    data = load_patients()
    profiles = data.get(username, [])
    profile["id"] = secrets.token_hex(6)
    profile["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    profiles.append(profile)
    data[username] = profiles
    save_patients(data)
    return profile

def delete_profile(username: str, profile_id: str):
    data = load_patients()
    data[username] = [p for p in data.get(username, []) if p["id"] != profile_id]
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

def add_history(username: str, profile: dict, joint: str, mode: str, angle: float = 0, count: int = 0, level: str = "正常", exercise_type: str = ""):
    records = load_history()
    records.append({
        "user": username,
        "patient_id": profile["id"],
        "patient_name": profile["name"],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "joint": joint,
        "mode": mode,
        "angle": angle,
        "count": count,
        "level": level,
        "exercise_type": exercise_type
    })
    save_history(records)

def profile_history(username: str, profile_id: str) -> list:
    return [
        r for r in load_history()
        if r.get("user") == username and r.get("patient_id") == profile_id
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

def get_lower_limb_points(landmarks, img_w, img_h, side="right"):
    pts = {}
    if side == "right":
        pts["hip"]   = [int(landmarks[24].x * img_w), int(landmarks[24].y * img_h)]
        pts["knee"]  = [int(landmarks[26].x * img_w), int(landmarks[26].y * img_h)]
        pts["ankle"] = [int(landmarks[28].x * img_w), int(landmarks[28].y * img_h)]
    else:
        pts["hip"]   = [int(landmarks[23].x * img_w), int(landmarks[23].y * img_h)]
        pts["knee"]  = [int(landmarks[25].x * img_w), int(landmarks[25].y * img_h)]
        pts["ankle"] = [int(landmarks[27].x * img_w), int(landmarks[27].y * img_h)]
    return pts

def draw_single_arm(frame, pts):
    cv2.line(frame, pts["shoulder"], pts["elbow"], (128, 0, 255), 3)
    cv2.line(frame, pts["elbow"], pts["wrist"], (128, 0, 255), 3)
    for k in ("shoulder", "elbow", "wrist"):
        cv2.circle(frame, pts[k], 6, (255, 255, 255), -1)

def draw_single_leg(frame, pts):
    cv2.line(frame, pts["hip"], pts["knee"], (128, 0, 255), 3)
    cv2.line(frame, pts["knee"], pts["ankle"], (128, 0, 255), 3)
    for k in ("hip", "knee", "ankle"):
        cv2.circle(frame, pts[k], 6, (255, 255, 255), -1)

def status_of(angle):
    if angle > REF_MAX:
        return f"🟡 超过参考上限{REF_MAX}°", (0, 255, 255), "warning", "超过上限"
    if angle >= NORMAL_MIN:
        return "✅ 屈曲角度正常", (0, 255, 0), "success", "正常"
    if angle >= MILD_MIN:
        return "⚠️ 轻度屈曲受限", (0, 165, 255), "warning", "轻度受限"
    return "❗ 明显屈曲受限", (0, 0, 255), "error", "明显受限"

# ===================== 核心视频处理器 =====================
class RealTimeProcessor(VideoProcessorBase):
    def __init__(self, mode, selected_side="left"):
        # 【性能优化 1】model_complexity=0 开启极速模型，极大加快处理速度
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mode = mode
        self.side = selected_side
        self.current_angle = 0.0
        self.max_angle = 0.0
        self.counter = 0
        self.stage = None
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.landmark_style = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
        self.connection_style = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 【性能优化 2】将进来的画面瞬间压缩到 320x240，运算量降低 80%，告别卡顿
        img = cv2.resize(img, (320, 240))
        img = auto_white_balance(img)
        img_h, img_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 1. 肘关节康复 (修复乱码：纯英文+数字)
            if self.mode == "elbow":
                pts = get_upper_limb_points(landmarks, img_w, img_h, self.side)
                raw_angle = calculate_raw_angle(pts["shoulder"], pts["elbow"], pts["wrist"])
                self.current_angle = raw_to_flexion(raw_angle)
                if self.current_angle > self.max_angle: self.max_angle = self.current_angle

                draw_single_arm(img, pts)
                cv2.putText(img, f"{self.current_angle} deg", (pts["elbow"][0] + 15, pts["elbow"][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(img, f"Current: {self.current_angle} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Max: {self.max_angle} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                text, color, _, _ = status_of(self.current_angle)
                status_en = "Normal" if "正常" in text else ("Limited" if "受限" in text else "Over")
                cv2.putText(img, f"Status: {status_en}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 2. 膝关节康复
            elif self.mode == "knee":
                pts = get_lower_limb_points(landmarks, img_w, img_h, self.side)
                raw_angle = calculate_raw_angle(pts["hip"], pts["knee"], pts["ankle"])
                self.current_angle = raw_to_flexion(raw_angle)
                if self.current_angle > self.max_angle: self.max_angle = self.current_angle

                draw_single_leg(img, pts)
                cv2.putText(img, f"{self.current_angle} deg", (pts["knee"][0] + 15, pts["knee"][1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(img, f"Current: {self.current_angle} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Max: {self.max_angle} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                text, color, _, _ = status_of(self.current_angle)
                status_en = "Normal" if "正常" in text else ("Limited" if "受限" in text else "Over")
                cv2.putText(img, f"Status: {status_en}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 3. 俯卧撑
            elif self.mode == "pushup":
                def get_pt(idx): return [int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)]
                shoulder, elbow, wrist = get_pt(11), get_pt(13), get_pt(15)
                hip, ankle = get_pt(23), get_pt(27)
                
                elbow_angle = calculate_raw_angle(shoulder, elbow, wrist)
                body_angle = calculate_raw_angle(shoulder, hip, ankle)
                
                if elbow_angle > 160: self.stage = "UP"
                if elbow_angle < 90 and self.stage == "UP":
                    self.stage = "DOWN"
                    self.counter += 1

                cv2.rectangle(img, (0, 0), (150, 60), (245, 117, 16), -1)
                cv2.putText(img, 'REPS', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.putText(img, str(self.counter), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                cv2.putText(img, 'STAGE', (80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                if self.stage: cv2.putText(img, self.stage, (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                if body_angle < 150 or body_angle > 195:
                    cv2.putText(img, "Keep body straight!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 4. 深蹲
            elif self.mode == "squat":
                def get_pt(idx): return [int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)]
                l_shoulder, r_shoulder = get_pt(11), get_pt(12)
                l_hip, r_hip = get_pt(23), get_pt(24)
                l_knee, r_knee = get_pt(25), get_pt(26)
                l_ankle, r_ankle = get_pt(27), get_pt(28)

                knee_angle = calculate_raw_angle(l_hip, l_knee, l_ankle)
                knee_dist = abs(l_knee[0] - r_knee[0])
                ankle_dist = abs(l_ankle[0] - r_ankle[0])

                if knee_angle > 160: self.stage = "UP"
                if knee_angle < 110 and self.stage == "UP":
                    self.stage = "DOWN"
                    self.counter += 1

                cv2.rectangle(img, (0, 0), (150, 60), (50, 150, 50), -1)
                cv2.putText(img, 'SQUATS', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                cv2.putText(img, str(self.counter), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
                cv2.putText(img, 'STAGE', (80, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                if self.stage: cv2.putText(img, self.stage, (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                
                if self.stage == "DOWN" and knee_dist < (ankle_dist * 0.8):
                     cv2.putText(img, "Knees caving in!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif self.stage == "DOWN" and knee_angle > 130:
                     cv2.putText(img, "Go deeper!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            self.mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_style,
                connection_drawing_spec=self.connection_style
            )

        # 处理完后放大回 640x480 以便前端显示清晰
        img = cv2.resize(img, (640, 480))
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
        <h1>康测 · 居家康复与运动监测系统</h1>
        <p>关节活动度评估 · 日常运动姿态纠错</p>
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
                    st.session_state.page = "👤 用户信息"
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

# ===================== 用户信息页 =====================
def patient_page():
    st.markdown(f"""
    <div class="hero">
        <h1>用户基本信息</h1>
        <p>账号：{st.session_state.username}　|　信息用于关联测量与运动记录</p>
    </div>
    """, unsafe_allow_html=True)

    profiles = get_user_profiles(st.session_state.username)

    if profiles:
        st.subheader("选择用户")
        options = {f"{p['name']}（{p.get('gender','')} / {p.get('age','')}岁）": p for p in profiles}
        choice = st.selectbox("当前操作的用户", list(options.keys()))
        selected = options[choice]

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 选定该用户并进入", use_container_width=True):
                st.session_state.current_patient = selected
                st.session_state.page = "📐 康复测量"
                st.rerun()
        with col2:
            if st.button("🗑️ 删除该用户", use_container_width=True):
                delete_profile(st.session_state.username, selected["id"])
                if st.session_state.get("current_patient", {}).get("id") == selected["id"]:
                    st.session_state.pop("current_patient", None)
                st.success("已删除")
                st.rerun()
        st.markdown("---")

    st.subheader("新增用户" if profiles else "请填写用户基本信息")
    with st.form("patient_form", clear_on_submit=True):
        name = st.text_input("姓名 *")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("性别", ["男", "女", "其他"])
        with c2:
            age = st.number_input("年龄", min_value=0, max_value=120, value=30, step=1)
        with c3:
            affected = st.selectbox("患侧（选填）", ["无", "右侧", "左侧", "双侧"])
        
        diagnosis = st.text_input("诊断 / 备注（选填）", placeholder="如：右肘骨折术后 3 个月（普通健身可不填）")
        submitted = st.form_submit_button("保存并进入", use_container_width=True)

    if submitted:
        if not name.strip():
            st.error("姓名不能为空")
        else:
            profile = add_profile(st.session_state.username, {
                "name": name.strip(),
                "gender": gender,
                "age": int(age),
                "affected": affected,
                "diagnosis": diagnosis.strip() if diagnosis.strip() else "无",
            })
            st.session_state.current_patient = profile
            st.session_state.page = "📐 康复测量"
            st.success(f"已保存用户：{profile['name']}")
            st.rerun()

# ===================== 拍照模式 =====================
def photo_mode(selected_side, joint_type):
    st.subheader(f"📷 拍照测量（{joint_type}）")
    photo = st.camera_input("Take a photo")
    if not photo:
        return

    img_np = np.array(Image.open(photo))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = auto_white_balance(img_bgr)
    h, w = img_bgr.shape[:2]

    img_rgb_wb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0, min_detection_confidence=0.5) as pose:
        res = pose.process(img_rgb_wb)

    if not res.pose_landmarks:
        st.error("未识别到人体关键点，请摆正姿势，完整入镜")
        return

    if joint_type == "肘关节":
        pts = get_upper_limb_points(res.pose_landmarks.landmark, w, h, selected_side)
        raw_angle = calculate_raw_angle(pts["shoulder"], pts["elbow"], pts["wrist"])
        angle = raw_to_flexion(raw_angle)
        draw_single_arm(img_bgr, pts)
        text_pos = (pts["elbow"][0] + 15, pts["elbow"][1] - 15)
    else:
        pts = get_lower_limb_points(res.pose_landmarks.landmark, w, h, selected_side)
        raw_angle = calculate_raw_angle(pts["hip"], pts["knee"], pts["ankle"])
        angle = raw_to_flexion(raw_angle)
        draw_single_leg(img_bgr, pts)
        text_pos = (pts["knee"][0] + 15, pts["knee"][1] - 15)

    cv2.putText(img_bgr, f"{angle} deg", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

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
                    joint_type, "拍照", angle=angle, count=0, level=level_cn, exercise_type="")
        st.session_state.last_photo_key = photo_key
        st.caption("✅ 已自动保存本次测量到历史记录")

# ===================== 视频实时模式 =====================
def video_mode(selected_side, joint_type):
    st.subheader(f"🎥 实时动态测量（{joint_type}）")
    st.info("活动关节，画面左上角实时显示当前/最大屈曲角度与评估状态。")
    
    mode_key = "elbow" if joint_type == "肘关节" else "knee"

    if ("processor" not in st.session_state 
        or getattr(st.session_state.processor, "mode", None) != mode_key
        or getattr(st.session_state.processor, "side", None) != selected_side):
        st.session_state.processor = RealTimeProcessor(mode=mode_key, selected_side=selected_side)
    
    current_processor = st.session_state.processor

    ctx = webrtc_streamer(
        key=f"{mode_key}-real-time-{selected_side}",
        video_processor_factory=lambda: current_processor,
        # 【性能优化 3】限制前端传输的视频分辨率为 320x240，减小网络传输负担
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        rtc_configuration=RTC_CONFIG,
        async_processing=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 重置本轮最大角度", use_container_width=True):
            current_processor.max_angle = 0.0
            st.success("最大角度已重置")
    with col2:
        if st.button("💾 保存本轮最大角度", use_container_width=True):
            if current_processor.max_angle <= 0:
                st.warning("当前最大角度为 0，尚未测到有效数据")
            else:
                _, _, _, level_cn = status_of(current_processor.max_angle)
                add_history(st.session_state.username, st.session_state.current_patient,
                            joint_type, "视频", angle=current_processor.max_angle, count=0, level=level_cn, exercise_type="")
                st.success(f"已保存：{current_processor.max_angle}°（{level_cn}）")

    if ctx.video_processor:
        st.write(f"本轮最大屈曲角度：{ctx.video_processor.max_angle} °")
        text, _, level, _ = status_of(ctx.video_processor.max_angle)
        getattr(st, level)(text)

# ===================== 康复测量页 =====================
def measure_page():
    profile = st.session_state.get("current_patient")
    if not profile:
        st.warning("尚未选择用户，请先填写用户基本信息")
        if st.button("前往用户信息"):
            st.session_state.page = "👤 用户信息"
            st.rerun()
        return

    st.info(f"👤 当前用户：**{profile['name']}**（{profile.get('gender','')} / {profile.get('age','')}岁 / 患侧：{profile.get('affected','')}）"
            + (f"　诊断：{profile['diagnosis']}" if profile.get('diagnosis') else ""))
    if st.button("切换用户"):
        st.session_state.page = "👤 用户信息"
        st.rerun()

    with st.expander("📖 操作引导：如何正确站立与测量（点击展开/收起）", expanded=False):
        st.markdown("""
        **① 站立姿势**
        - **侧对摄像头站立**，身体与镜头垂直，双脚与肩同宽。
        - **待测部位完整入镜**：测肘关节时手臂不可出画；测膝关节时需保证髋部到脚踝完整入画。
        - 摄像头高度**与胸口齐平**，距离约 **1.5～2 米**。
        
        **② 光线与环境**
        - **面向光源站立，避免背光**。背景简洁，穿短袖/短裤避免遮挡。
        
        **③ 测量动作**
        - **伸直位**：自然下垂伸直，应接近 **0°**。
        - **弯曲**：缓慢屈曲到最大角度并保持 1～2 秒。
        """)

    joint_option = st.selectbox("请选择待测部位", ["肘关节", "膝关节"])
    side_option = st.selectbox("请选择待测侧别", ["右侧", "左侧"])
    side_key = "right" if side_option == "右侧" else "left"

    tab1, tab2 = st.tabs(["📷 拍照模式", "🎥 实时视频模式"])
    with tab1:
        photo_mode(side_key, joint_option)
    with tab2:
        video_mode(side_key, joint_option)

# ===================== 日常运动页 =====================
def exercise_page():
    profile = st.session_state.get("current_patient")
    if not profile:
        st.warning("请先前往『用户信息』选择一个用户，再进行运动记录。")
        if st.button("前往用户信息"):
            st.session_state.page = "👤 用户信息"
            st.rerun()
        return

    st.markdown("""
    <div class="hero">
        <h1>日常居家运动监测</h1>
        <p>AI 姿态纠错 · 自动计数 · 避免运动损伤</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.info(f"👤 当前运动者：**{profile['name']}**（{profile.get('age','')}岁）。请完整露在镜头中，系统将自动为您计数并纠错。")
    st.warning("⚠️ **免责声明**：本模块仅作为日常健身姿态参考，不能替代专业教练或医生的指导。若运动中出现疼痛，请立刻停止。")

    exercise_choice = st.selectbox("请选择运动项目", ["俯卧撑 (Push-up)", "深蹲 (Squat)"])
    mode_key = "pushup" if "俯卧撑" in exercise_choice else "squat"
    ex_name = "俯卧撑" if mode_key == "pushup" else "深蹲"

    if ("processor" not in st.session_state 
        or getattr(st.session_state.processor, "mode", None) != mode_key):
        st.session_state.processor = RealTimeProcessor(mode=mode_key)
    
    current_processor = st.session_state.processor

    ctx = webrtc_streamer(
        key=f"{mode_key}-real-time",
        video_processor_factory=lambda: current_processor,
        # 【性能优化 3】限制前端传输的视频分辨率为 320x240
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
        rtc_configuration=RTC_CONFIG,
        async_processing=True
    )

    st.markdown("---")
    st.subheader("📝 运动记录保存")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 重置当前计数", use_container_width=True):
            current_processor.counter = 0
            current_processor.stage = None
            st.success("当前计数已归零")
    
    with col2:
        if st.button("💾 结束并保存记录", use_container_width=True):
            if current_processor.counter <= 0:
                st.warning("当前次数为 0，无需保存")
            else:
                add_history(st.session_state.username, st.session_state.current_patient,
                            joint="无", mode=ex_name, angle=0, count=current_processor.counter, level="正常", exercise_type=ex_name)
                st.success(f"✅ 成功保存！本次做了 {current_processor.counter} 个{ex_name}")
                current_processor.counter = 0 
                current_processor.stage = None

# ===================== 趋势图 =====================
def render_trend_chart(df: pd.DataFrame):
    if df.empty:
        return

    chart_df = df.copy()
    chart_df["时间_dt"] = pd.to_datetime(chart_df["时间"])
    chart_df = chart_df.sort_values("时间_dt")

    fig = go.Figure()
    
    sub_right = chart_df[chart_df["侧别"] == "右侧"] if "侧别" in chart_df.columns else pd.DataFrame()
    if not sub_right.empty:
        fig.add_trace(go.Scatter(
            x=sub_right["时间_dt"], y=sub_right["屈曲角度(°)"],
            mode="lines+markers", name="右侧",
            line=dict(color="#1c4a86", width=2.5),
            marker=dict(size=8, line=dict(color="white", width=1)),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>角度：%{y:.1f}°<extra></extra>",
        ))

    sub_left = chart_df[chart_df["侧别"] == "左侧"] if "侧别" in chart_df.columns else pd.DataFrame()
    if not sub_left.empty:
        fig.add_trace(go.Scatter(
            x=sub_left["时间_dt"], y=sub_left["屈曲角度(°)"],
            mode="lines+markers", name="左侧",
            line=dict(color="#e07b00", width=2.5),
            marker=dict(size=8, line=dict(color="white", width=1)),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>角度：%{y:.1f}°<extra></extra>",
        ))

    fig.add_hline(y=120, line_dash="dash", line_color="#2e9e5b", annotation_text="正常下限 120°", annotation_position="top left")
    fig.add_hline(y=150, line_dash="dot", line_color="#c9a400", annotation_text="参考上限 150°", annotation_position="bottom left")

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
    profile = st.session_state.get("current_patient")
    if not profile:
        st.warning("请先前往『用户信息』选择用户")
        if st.button("前往用户信息"):
            st.session_state.page = "👤 用户信息"
            st.rerun()
        return

    st.markdown(f"""
    <div class="hero">
        <h1>历史测量与运动记录</h1>
        <p>用户：{profile['name']}　|　{profile.get('gender','')} / {profile.get('age','')}岁</p>
    </div>
    """, unsafe_allow_html=True)

    records = profile_history(st.session_state.username, profile["id"])
    if not records:
        st.info("暂无历史记录。完成一次测量或运动后，记录会出现在这里。")
        return

    df = pd.DataFrame(records)
    if "joint" not in df.columns: df["joint"] = "肘关节"
    if "count" not in df.columns: df["count"] = 0

    tab1, tab2 = st.tabs(["📐 关节康复记录", "🏃 运动打卡记录"])

    with tab1:
        df_angle = df[df["mode"].isin(["拍照", "视频"])].copy()
        if df_angle.empty:
            st.info("暂无关节康复记录")
        else:
            df_angle = df_angle[["time", "side", "joint", "mode", "angle", "level"]].rename(columns={
                "time": "时间", "side": "侧别", "joint": "部位", "mode": "模式", "angle": "屈曲角度(°)", "level": "评估"
            })
            df_angle = df_angle.sort_values("时间", ascending=False).reset_index(drop=True)

            st.subheader("📈 趋势分析")
            joint_filter = st.radio("选择查看部位", ["全部", "肘关节", "膝关节"], horizontal=True)
            chart_df = df_angle if joint_filter == "全部" else df_angle[df_angle["部位"] == joint_filter]
            render_trend_chart(chart_df)

            if len(df_angle) >= 2:
                recent = df_angle.head(5)
                older = df_angle.iloc[5:10]
                if not older.empty:
                    delta = recent["屈曲角度(°)"].mean() - older["屈曲角度(°)"].mean()
                    if delta > 3:
                        st.success(f"📈 近期平均角度较之前提升 {delta:.1f}°，康复趋势向好")
                    elif delta < -3:
                        st.warning(f"📉 近期平均角度较之前下降 {abs(delta):.1f}°，建议关注")
                    else:
                        st.info(f"➡️ 近期角度变化平稳（{delta:+.1f}°）")

            c1, c2, c3 = st.columns(3)
            c1.metric("记录总数", len(df_angle))
            c2.metric("最大角度", f"{df_angle['屈曲角度(°)'].max():.1f}°")
            c3.metric("最近一次", df_angle["时间"].iloc[0][5:16])

            st.dataframe(df_angle, use_container_width=True, hide_index=True)

    with tab2:
        df_ex = df[df["mode"].isin(["俯卧撑", "深蹲"])].copy()
        if df_ex.empty:
            st.info("暂无运动打卡记录，快去『日常运动』模块锻炼吧！")
        else:
            df_ex = df_ex[["time", "mode", "count"]].rename(columns={
                "time": "时间", "mode": "运动项目", "count": "完成次数"
            })
            df_ex = df_ex.sort_values("时间", ascending=False).reset_index(drop=True)
            
            c1, c2 = st.columns(2)
            c1.metric("累计运动组数", len(df_ex))
            c2.metric("累计总次数", int(df_ex["完成次数"].sum()))
            
            st.dataframe(df_ex, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.download_button(
        "⬇️ 导出全部 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"history_{profile['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ===================== 主入口 =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "📐 康复测量"

if not st.session_state.logged_in:
    login_page()
    st.stop()

st.markdown(f"""
<div class="hero">
    <h1>康测 · 一体化居家姿态监测</h1>
    <p>当前用户：{st.session_state.username}　|　康复评估与日常运动纠错</p>
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
        st.session_state.page = "📐 康复测量"
        st.rerun()
    st.markdown("---")

    nav_options = ["📐 康复测量", "🏃 日常运动", "👤 用户信息", "📜 历史记录"]
    page = st.radio(
        "功能导航", nav_options,
        index=nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0,
        label_visibility="collapsed",
    )
    st.session_state.page = page

    st.markdown("---")
    st.markdown("> 本工具仅作居家康复初筛参考，不能替代医院专业评估。")

if page == "👤 用户信息":
    patient_page()
elif page == "📜 历史记录":
    history_page()
elif page == "🏃 日常运动":
    exercise_page()
else:
    measure_page()
