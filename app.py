from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify, send_file
import sqlite3
import cv2
import threading
import mediapipe as mp
import time
import string
from uuid import uuid4
from functools import wraps
import json
from gtts import gTTS
import io

# ---------------- Config ----------------
app = Flask(__name__)
app.secret_key = "hushtone_secret_key_change_this"
DB_NAME = "hushtone_users.db"

# Hardcoded admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# ---------------- Load Languages ----------------
with open("languages.json", encoding="utf-8") as f:
    translations = json.load(f)

# ---------------- Helpers ----------------
def get_db_conn():
    return sqlite3.connect(DB_NAME)

# ---------------- DB Init ----------------
def init_db():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password TEXT,
            name TEXT,
            age INTEGER,
            city TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS gesture_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            guest_id TEXT,
            gesture TEXT,
            action_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS gesture_meanings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gesture_name TEXT NOT NULL,
            custom_meaning TEXT NOT NULL,
            language TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending',
            reviewed_by TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)
        conn.commit()

# Initialize DB
init_db()

# Ensure 'reviewed_by' column exists (for old DBs)
with get_db_conn() as conn:
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE gesture_meanings ADD COLUMN reviewed_by TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists, do nothing
        pass

# ---------------- Globals ----------------
cap = None
frame = None
recognition_running = False
gesture_text = ""
gesture_image = None
current_user = None
current_user_id = None
current_guest_id = None

# ---------------- Mediapipe ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# ---------------- Gesture dictionaries ----------------
gesture_dict = {
    "fist": "Stop",
    "open": "Hello",
    "thumbs_up": "Yes",
    "peace": "Peace",
    "call_me": "Call me",
    "ok": "OK",
    "pointing": "Look here",
    "rock_on": "Rock on!"
}
for i in range(0, 6):
    gesture_dict[f"number_{i}"] = str(i)
for letter in string.ascii_uppercase:
    gesture_dict[f"alphabet_{letter}"] = letter

gesture_images = {}
for i in range(0, 6):
    gesture_images[f"number_{i}"] = f"numbers/{i}.jpg"
for letter in string.ascii_uppercase:
    gesture_images[f"alphabet_{letter}"] = f"alphabets/{letter}.jpg"

smart_suggestions = {
    "A": ["Apple", "Ant", "Airplane"],
    "B": ["Ball", "Banana", "Book"],
    "C": ["Cat", "Car", "Cup"],
    "D": ["Dog", "Door", "Duck"],
    "E": ["Elephant", "Egg", "Eagle"]
}

# ---------------- Helpers ----------------
def get_db_conn():
    return sqlite3.connect(DB_NAME)

def get_user_action_text(user_id, gesture):
    """Return the approved custom meaning for this user if exists, otherwise default"""
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT custom_meaning FROM gesture_meanings
            WHERE gesture_name=? AND user_id=? AND status='approved'
            ORDER BY timestamp DESC LIMIT 1
        """, (gesture, user_id))
        row = cur.fetchone()
        if row:
            return row[0]  # return the user's approved meaning
    # fallback to default
    return gesture_dict.get(gesture, "")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session or session.get('role') == 'admin':
            flash("Login first.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            session.pop('user', None)
            session.pop('user_id', None)
            flash("Admin access required.")
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated

# ---------------- Gesture Recognition ----------------
def finger_states(hand):
    fingers = []
    fingers.append(1 if hand.landmark[4].x < hand.landmark[3].x else 0)
    fingers.append(1 if hand.landmark[8].y < hand.landmark[6].y else 0)
    fingers.append(1 if hand.landmark[12].y < hand.landmark[10].y else 0)
    fingers.append(1 if hand.landmark[16].y < hand.landmark[14].y else 0)
    fingers.append(1 if hand.landmark[20].y < hand.landmark[18].y else 0)
    return fingers

def recognize_gesture(hand):
    fingers = finger_states(hand)
    total_fingers = sum(fingers)
    if fingers == [0,0,0,0,0]: return "fist"
    if fingers == [1,1,1,1,1]: return "open"
    if fingers == [1,0,0,0,0]: return "thumbs_up"
    if fingers == [0,1,1,0,0]: return "peace"
    if fingers == [0,1,0,0,1]: return "rock_on"
    if fingers == [0,1,0,0,0]: return "pointing"
    if fingers == [1,0,0,0,1]: return "call_me"
    if total_fingers <= 5: return f"number_{total_fingers}"
    return None

last_gesture = None
gesture_cooldown = 0.5
last_time = 0

def store_gesture_to_db(user_id, guest_id, gesture, action_text):
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO gesture_history (user_id, guest_id, gesture, action_text, timestamp) VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            (user_id, guest_id, gesture, action_text)
        )
        conn.commit()

def gesture_thread():
    global cap, frame, recognition_running, gesture_text, last_gesture, last_time, gesture_image
    while recognition_running and cap and cap.isOpened():
        success, img = cap.read()
        if not success:
            time.sleep(0.01)
            continue
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        gesture_text = None
        gesture_image = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks)
                if gesture:
                    current_time = time.time()
                    if gesture != last_gesture or (current_time - last_time) > gesture_cooldown:
                        gesture_text = gesture   # store gesture key like "open"
                        uid = current_user_id  # current logged-in user
                        action_text = get_user_action_text(uid, gesture) if uid else gesture_dict.get(gesture, "")
                        last_gesture = gesture
                        last_time = current_time
                        uid = current_user_id if current_user_id else None
                        gid = current_guest_id if current_guest_id else None
                        store_gesture_to_db(uid, gid, gesture, action_text)
                        if gesture in gesture_images:
                            gesture_image = gesture_images[gesture]
        frame = img

# ---------------- Video Generator ----------------
def gen_frames():
    global frame
    while True:
        if frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------- Routes ----------------
@app.route('/')
def home(): 
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id,password FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            if row and row[1] == password:
                session['user'] = username
                session['user_id'] = row[0]
                session.pop('role', None)
                flash("Logged in successfully.")
                return redirect(url_for('main'))
            else:
                flash("Invalid username or password")
    return render_template("login.html")

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method=='POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()
        try:
            with get_db_conn() as conn:
                cur = conn.cursor()
                cur.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                            (username,email,password))
                conn.commit()
            flash("Account created! Login now.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.")
    return render_template("signup.html")

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for('login'))

# ---------------- Main and Page Routes ----------------
@app.route('/main')
@login_required
def main():
    return render_template("main.html", username=session['user'])

@app.route('/account', methods=['GET','POST'])
@login_required
def account():
    user_id = session['user_id']
    username = session['user']

    with get_db_conn() as conn:
        cur = conn.cursor()
        if request.method == 'POST':
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip()
            age = request.form.get('age', '').strip()
            city = request.form.get('city', '').strip()
            try:
                cur.execute("""
                    UPDATE users 
                    SET name=?, email=?, age=?, city=?
                    WHERE id=?
                """, (name, email, age or None, city, user_id))
                conn.commit()
                flash("Profile updated successfully!")
            except sqlite3.IntegrityError:
                flash("Email already exists. Try another.")
        cur.execute("SELECT name,email,age,city FROM users WHERE id=?", (user_id,))
        user_data = cur.fetchone()
        user_info = {
            'name': user_data[0] or '',
            'email': user_data[1] or '',
            'age': user_data[2] or '',
            'city': user_data[3] or ''
        }
    return render_template("account.html", username=username, **user_info)

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    user_id = session['user_id']
    current = request.form.get('current_password', '').strip()
    new = request.form.get('new_password', '').strip()
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE id=?", (user_id,))
        stored_password = cur.fetchone()[0]
        if stored_password != current:
            flash("Current password is incorrect!")
        elif len(new) < 4:
            flash("New password must be at least 4 characters.")
        else:
            cur.execute("UPDATE users SET password=? WHERE id=?", (new, user_id))
            conn.commit()
            flash("Password changed successfully!")
    return redirect(url_for('account'))

@app.route('/history')
@login_required
def history():
    username = session['user']
    user_id = session['user_id']
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT gesture, timestamp FROM gesture_history WHERE user_id=? ORDER BY timestamp DESC LIMIT 20", (user_id,))
        rows = cur.fetchall()
    return render_template("history.html", title="Your Gesture History", rows=rows, username=username)

@app.route('/guidelines')
@login_required
def guidelines():
    return render_template("guidelines.html", username=session['user'])

# ---------------- Submit Gesture Meaning ----------------
@app.route('/submit_gesture_meaning', methods=['GET', 'POST'])
@login_required
def submit_gesture_meaning():
    if request.method == 'POST':
        gesture_name = request.form.get('gesture_name')
        custom_meaning = request.form.get('custom_meaning')
        language = request.form.get('language')
        user_id = session.get('user_id')

        if not gesture_name or not custom_meaning or not language:
            flash("All fields are required.")
            return redirect(url_for('submit_gesture_meaning'))

        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO gesture_meanings
                (gesture_name, custom_meaning, language, user_id, status)
                VALUES (?, ?, ?, ?, 'pending')
            """, (gesture_name, custom_meaning, language, user_id))
            conn.commit()

        flash("Gesture meaning submitted for admin approval.")
        return redirect(url_for('submit_gesture_meaning'))

    return render_template("submit_gesture_meaning.html")

# ---------------- View My Submissions ----------------
@app.route('/my_submissions')
@login_required
def my_submissions():
    user_id = session['user_id']
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT gesture_name, custom_meaning, language, status, timestamp
            FROM gesture_meanings
            WHERE user_id=?
            ORDER BY timestamp DESC
        """, (user_id,))
        submissions = cur.fetchall()
    return render_template("my_submissions.html", submissions=submissions)

@app.route('/admin_login', methods=['GET','POST'])
def admin_login():
    session.pop('_flashes', None)  # Clear old flash messages

    if request.method == 'POST':
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['user'] = ADMIN_USERNAME
            session['role'] = 'admin'
            flash("Admin logged in successfully!")
            return redirect(url_for('admin_dashboard'))  # ✅ only on success
        else:
            flash("Invalid admin credentials.")  # ✅ only on failure

    return render_template("admin_login.html")

# Admin Dashboard — just shows 3 buttons/boxes
@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')


# 1️⃣ Manage Users page
@app.route('/admin/users')
@admin_required
def manage_users():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, email, name, age, city FROM users ORDER BY id DESC")
        users = cur.fetchall()
    return render_template("admin_users.html", users=users)


# 2️⃣ View History page
@app.route('/admin/history')
@admin_required
def view_history():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT gh.id, u.username, gh.gesture, gh.action_text, gh.timestamp 
            FROM gesture_history gh
            LEFT JOIN users u ON u.id = gh.user_id
            ORDER BY gh.timestamp DESC LIMIT 50
        """)
        history = cur.fetchall()
    return render_template("admin_history.html", history=history)


# 3️⃣ Gesture Meaning Approvals page
@app.route('/admin/gesture-approvals')
@admin_required
def gesture_approvals():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT gm.id, u.username, gm.gesture_name, gm.custom_meaning, gm.language, gm.timestamp
            FROM gesture_meanings gm
            JOIN users u ON gm.user_id = u.id
            WHERE gm.status='pending'
            ORDER BY gm.timestamp DESC
        """)
        pending_meanings = cur.fetchall()
    return render_template("admin_approvals.html", pending_meanings=pending_meanings)


# Approve a pending gesture meaning
@app.route('/admin/approve/<int:meaning_id>')
@admin_required
def approve_meaning(meaning_id):
    admin_username = session.get('user')  # current admin
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE gesture_meanings 
            SET status='approved', reviewed_by=?
            WHERE id=?
        """, (admin_username, meaning_id))
        conn.commit()
    flash("Gesture meaning approved.")
    return redirect(url_for('gesture_approvals'))

# Reject a pending gesture meaning
@app.route('/admin/reject/<int:meaning_id>')
@admin_required
def reject_meaning(meaning_id):
    admin_username = session.get('user')  # current admin
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE gesture_meanings 
            SET status='rejected', reviewed_by=?
            WHERE id=?
        """, (admin_username, meaning_id))
        conn.commit()
    flash("Gesture meaning rejected.")
    return redirect(url_for('gesture_approvals'))

# Delete a user
@app.route('/admin/delete_user/<int:user_id>')
@admin_required
def delete_user(user_id):
    with get_db_conn() as conn:
        cur = conn.cursor()
        # Optional: prevent admin from deleting themselves if needed
        cur.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    flash("User deleted successfully.")
    return redirect(url_for('manage_users'))

# Clear all gesture history
@app.route('/admin/clear_history')
@admin_required
def clear_history():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM gesture_history")
        conn.commit()
    flash("All gesture history cleared.")
    return redirect(url_for('view_history'))

# View all gesture submissions (approved, rejected, pending)
@app.route('/admin/all_submissions')
@admin_required
def admin_all_submissions():
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT gm.id, u.username, gm.gesture_name, gm.custom_meaning, gm.language, gm.status, gm.timestamp
            FROM gesture_meanings gm
            JOIN users u ON gm.user_id = u.id
            ORDER BY gm.timestamp DESC
        """)
        submissions = cur.fetchall()
    return render_template("admin_all_submissions.html", submissions=submissions)


# ---------------- Video feed ----------------
@app.route('/video_feed')
def video_feed(): 
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition')
def start_recognition():
    global cap, recognition_running, current_user, current_user_id, current_guest_id
    if 'user' in session and session.get('role') != 'admin':
        current_user = session['user']
        current_user_id = session.get('user_id')
        current_guest_id = None
    else:
        current_user = None
        current_user_id = None
        if 'guest_id' not in session:
            session['guest_id'] = str(uuid4())
        current_guest_id = session['guest_id']
    if not recognition_running:
        cap=cv2.VideoCapture(0)
        recognition_running=True
        threading.Thread(target=gesture_thread, daemon=True).start()
    return ('',204)

@app.route('/stop_recognition')
def stop_recognition():
    global cap, recognition_running, current_user, current_user_id, current_guest_id
    recognition_running=False
    if cap: cap.release(); cap=None
    current_user=current_user_id=current_guest_id=None
    return ('',204)

@app.route('/gesture_status')
def gesture_status():
    global gesture_text, gesture_image
    uid = session.get('user_id')
    gid = session.get('guest_id')
    history = []

    with get_db_conn() as conn:
        cur = conn.cursor()
        # Fetch last 10 gestures for this user or guest
        if uid:
            cur.execute(
                "SELECT gesture, action_text, timestamp FROM gesture_history WHERE user_id=? ORDER BY timestamp DESC LIMIT 10",
                (uid,)
            )
        elif gid:
            cur.execute(
                "SELECT gesture, action_text, timestamp FROM gesture_history WHERE guest_id=? ORDER BY timestamp DESC LIMIT 10",
                (gid,)
            )
        else:
            cur.execute(
                "SELECT gesture, action_text, timestamp FROM gesture_history ORDER BY timestamp DESC LIMIT 0"
            )
        rows = cur.fetchall()
        history = [{"gesture": r[0], "action_text": r[1], "ts": r[2]} for r in rows]

    translated_text = ""
    if gesture_text:
        # --- Step 1: Check for user-specific approved custom meaning ---
        if uid:
            with get_db_conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT custom_meaning 
                    FROM gesture_meanings
                    WHERE gesture_name=? AND user_id=? AND status='approved'
                    ORDER BY timestamp DESC LIMIT 1
                """, (gesture_text, uid))
                row = cur.fetchone()
                if row:
                    translated_text = row[0]  # Use custom meaning for this user

        # --- Step 2: If no custom meaning, fallback to translations/default ---
        if not translated_text:
            lang = request.args.get("lang", "en")
            lang_key = lang.split("-")[0] if "-" in lang else lang
            translated_text = translations.get(lang_key, {}).get(gesture_text, gesture_dict.get(gesture_text, ""))

    # Smart suggestions for alphabets remain the same
    suggestions = []
    if gesture_text and gesture_text.startswith("alphabet_"):
        letter = gesture_text.split("_")[-1]
        suggestions = smart_suggestions.get(letter.upper(), [])

    return jsonify({
        "gesture": gesture_text or "",
        "translated": translated_text or "",
        "image": "/static/" + gesture_image if gesture_image else "",
        "history": history,
        "suggestions": suggestions
    })

# ---------------- Server-side TTS ----------------
@app.route("/speak")
def speak():
    text = request.args.get("text", "")
    lang_code = request.args.get("lang", "en")
    if lang_code in ["ta", "ml", "hi", "en"]:
        tts = gTTS(text=text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return send_file(fp, mimetype="audio/mpeg")
    else:
        return jsonify({"status":"unsupported"})

# ---------------- Run ----------------
if __name__=="__main__":
    app.run(debug=True, threaded=True)
