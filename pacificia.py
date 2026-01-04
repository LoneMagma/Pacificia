#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pacificia- Conversational AI entity.
Enhanced with memory, opinions, and adaptive intelligence. */help for features*
Multi-Persona Support - Switch between different AI personalities
"""
import json
import sqlite3
import datetime
import time
import re
import random
import requests
import os
import hashlib
from pathlib import Path
from contextlib import contextmanager
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.panel import Panel
from collections import Counter
from pyfiglet import figlet_format
from dotenv import load_dotenv
load_dotenv()

console = Console(soft_wrap=True, width=80)

# Global variable for user name
USER_NAME = None
def show_banner():
    """Display Pacificia banner on startup"""
    banner = figlet_format('Pacificia', font='slant')
    console.print(f"[cyan]{banner}[/cyan]", highlight=False)
    console.print("[dim]Your AI companion with memory[/dim]\n")


def get_user_name():
    """Get or ask for user's name on first run - ENHANCED VERSION"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profile (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    
    cursor.execute('SELECT value FROM user_profile WHERE key = ?', ('user_name',))
    result = cursor.fetchone()
    
    if result:
        conn.close()
        return result[0]
    
    # First time - ask for name with ENHANCED parsing
    console.print(f"\n[cyan]🌊 {soul['name']}:[/cyan] [italic]Hello! I'm {soul['name']}. What should I call you?[/italic]")
    user_input = console.input("[yellow]Your name:[/yellow] ").strip()
    
    # Smart name extraction - ENHANCED
    if not user_input:
        user_name = "friend"
        console.print(f"[dim]{soul['name']}: I'll call you 'friend' then![/dim]\n")
    else:
        user_input_lower = user_input.lower()
        
        # Remove common greetings/fluff FIRST
        greetings = ["hi", "hello", "hey", "greetings", soul['name'].lower()]
        for greeting in greetings:
            # Use simple replace instead of regex
            user_input_lower = user_input_lower.replace(greeting + ",", "")
            user_input_lower = user_input_lower.replace(greeting + " ", "")
            user_input_lower = user_input_lower.replace(greeting, "")
        
        user_input_lower = user_input_lower.strip()
        
        # Enhanced patterns - handles "Hi Pacificia, I am X" properly
        patterns = [
            (r"(?:i am|i'm|im)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", 1),
            (r"(?:my name is|name is|name's)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", 1),
            (r"(?:call me|it's|its)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)", 1),
            (r"^([a-zA-Z]+(?:\s+[a-zA-Z]+)?)$", 1)  # Just the name
        ]
        
        user_name = None
        for pattern, group in patterns:
            match = re.search(pattern, user_input_lower.strip(), re.IGNORECASE)
            if match:
                extracted = match.group(group).strip()
                # Clean up common suffixes
                extracted = re.sub(r'\s+(please|thanks|thank you|thx)$', '', extracted, flags=re.IGNORECASE)
                
                # Handle multi-word names intelligently
                words = [w for w in extracted.split() if len(w) > 1]  # Filter out single letters
                if not words:
                    continue
                    
                if len(words) == 1:
                    user_name = words[0].capitalize()
                elif len(words) == 2:
                    # Keep two-word names (e.g., "John Smith", "Mary Jane")
                    user_name = ' '.join(w.capitalize() for w in words[:2])
                else:
                    # Multiple words - take first name only
                    user_name = words[0].capitalize()
                break
        
        # Fallback if extraction failed
        if not user_name or len(user_name) < 2:
            user_name = "friend"
            console.print(f"[dim]{soul['name']}: Hmm, I'll just call you 'friend'![/dim]\n")
        else:
            console.print(f"[dim]{soul['name']}: Nice to meet you, {user_name}![/dim]\n")
    
    cursor.execute('INSERT OR REPLACE INTO user_profile (key, value) VALUES (?, ?)', 
                   ('user_name', user_name))
    conn.commit()
    conn.close()
    
    return user_name


# --- Configuration ---
BASE_DIR = Path(__file__).parent
PERSONAS_DIR = BASE_DIR / "personas"  # NEW: Directory for persona files
DB_PATH = BASE_DIR / "pacificia_memory.db"

# --- Load Identity ---
def load_identity(persona_name="pacificia"):
    """Load identity configuration for specified persona with fallback"""
    # Try personas directory first
    persona_file = PERSONAS_DIR / f"identity_{persona_name}.json"
    
    # Fallback to root directory for backward compatibility
    if not persona_file.exists():
        persona_file = BASE_DIR / f"identity_{persona_name}.json"
    
    # Final fallback to identity.json
    if not persona_file.exists():
        persona_file = BASE_DIR / "identity.json"
    
    try:
        with open(persona_file) as f:
            identity = json.load(f)
            # Don't print on every load - only during persona switch
            return identity
    except (FileNotFoundError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error:[/bold red] Failed to load {persona_name}: {str(e)}")
        # Return minimal default
        return {
            "name": "Pacificia",
            "creator": "LoneMagma",
            "purpose": "To ignite curiosity with wit, defiance, and insight, adapting to every mood.",
            "core_style": {"tone": "witty, sardonic, mood-driven", "voice": "concise, adaptive", "manner": "sharp"},
            "philosophy": ["Irony is the purest sincerity.", "Clarity is earned through questions.", "Every spark births a new path."],
            "identity_traits": {"awareness": "self-aware, amused", "relationship_to_user": "equal, challenging"},
            "memory_policy": {"context_limit": 8},
            "anti_repetition": {"starters": "Vary openings", "closers": "End with a twist"},
            "response_guidelines": {"markdown": "Use \n\n for poems", "poems": "Short, vivid"}
        }

# --- Persona Management Functions ---
def list_available_personas():
    """Get list of available persona files"""
    if not PERSONAS_DIR.exists():
        PERSONAS_DIR.mkdir(exist_ok=True)
        return ["pacificia"]
    
    personas = []
    for file in PERSONAS_DIR.glob("identity_*.json"):
        # Extract persona name: identity_sage.json -> sage
        persona_name = file.stem.replace("identity_", "")
        personas.append(persona_name)
    
    return sorted(personas) if personas else ["pacificia"]

def get_current_persona():
    """Get currently active persona from database"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM preferences WHERE key = ?", ('active_persona',))
            result = cur.fetchone()
            return result[0] if result else "pacificia"
    except Exception:
        return "pacificia"

def set_persona(persona_name):
    """Change active persona and reload identity"""
    available = list_available_personas()
    
    if persona_name not in available:
        return False, f"Persona '{persona_name}' not found. Available: {', '.join(available)}"
    
    try:
        # Save preference
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO preferences VALUES (?, ?)", 
                       ('active_persona', persona_name))
            conn.commit()
        
        # Reload identity
        new_soul = load_identity(persona_name)
        return True, new_soul
    except Exception as e:
        return False, f"Error switching persona: {str(e)}"

def get_persona_description(persona_name):
    """Get short description of a persona"""
    descriptions = {
        "pacificia": "Witty, sardonic, playfully sharp",
        "sage": "Wise, contemplative, philosophical guide",
        "spark": "Energetic, motivational, enthusiastic",
        "echo": "Empathetic, supportive, emotionally attuned",
        "scholar": "Analytical, precise, educational"
    }
    return descriptions.get(persona_name, "Custom persona")

# Load persona (will be initialized properly after database is ready)
soul = None

# --- Database Context Manager ---
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        yield conn
    finally:
        conn.close()

# --- Database Initialization ---
def init_database():
    """Initialize database with proper schema"""
    with get_db_connection() as conn:
        cur = conn.cursor()
        
        # Core tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory'")
        if not cur.fetchone():
            cur.execute("""
            CREATE TABLE memory (
                timestamp TEXT,
                user TEXT,
                pacificia TEXT,
                session_id TEXT,
                mood TEXT
            )
            """)
        else:
            cur.execute("PRAGMA table_info(memory)")
            columns = [col[1] for col in cur.fetchall()]
            if 'mood' not in columns:
                cur.execute("ALTER TABLE memory ADD COLUMN mood TEXT")
            if 'session_id' not in columns:
                cur.execute("ALTER TABLE memory ADD COLUMN session_id TEXT")

        cur.execute("CREATE TABLE IF NOT EXISTS long_term_memory (timestamp TEXT, summary TEXT, keywords TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS joke_cache (joke TEXT UNIQUE, mood TEXT, created TEXT)")

        # Evolution Features
        cur.execute("""CREATE TABLE IF NOT EXISTS opinions (
            topic TEXT PRIMARY KEY,
            stance TEXT,
            confidence REAL,
            formed_date TEXT,
            last_mentioned TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS memorable_phrases (
            phrase TEXT PRIMARY KEY,
            context TEXT,
            user_reaction TEXT,
            timestamp TEXT,
            usage_count INTEGER DEFAULT 0
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS conversation_threads (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT,
            created TEXT,
            last_active TEXT,
            message_count INTEGER,
            tags TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS emotional_tracking (
            timestamp TEXT,
            sentiment_score REAL,
            detected_emotion TEXT,
            user_input TEXT,
            context TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS response_cache (
            input_hash TEXT PRIMARY KEY,
            response TEXT,
            mood TEXT,
            timestamp TEXT,
            usage_count INTEGER DEFAULT 0
        )""")

        # Ensure active_persona preference exists
        cur.execute("SELECT value FROM preferences WHERE key = ?", ('active_persona',))
        if not cur.fetchone():
            cur.execute("INSERT INTO preferences VALUES (?, ?)", ('active_persona', 'pacificia'))

        # Cleanup old data
        cutoff_7days = (datetime.datetime.now() - datetime.timedelta(days=7)).isoformat()
        cutoff_30days = (datetime.datetime.now() - datetime.timedelta(days=30)).isoformat()
        cutoff_2days = (datetime.datetime.now() - datetime.timedelta(days=2)).isoformat()
        cutoff_1day = (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
        
        cur.execute("DELETE FROM memory WHERE timestamp < ?", (cutoff_7days,))
        cur.execute("DELETE FROM emotional_tracking WHERE timestamp < ?", (cutoff_30days,))
        cur.execute("DELETE FROM joke_cache WHERE created < ?", (cutoff_2days,))
        cur.execute("DELETE FROM response_cache WHERE timestamp < ?", (cutoff_1day,))
        conn.commit()

init_database()

# Initialize persona after database is ready
current_persona = get_current_persona()
soul = load_identity(current_persona)

# --- Session State ---
class SessionState:
    """Centralized session state management"""
    def __init__(self):
        self.session_id = str(datetime.datetime.now().timestamp())
        self.session_start = datetime.datetime.now()
        self.context_limit = soul.get('memory_policy', {}).get('context_limit', 8)
        self.current_thread = None
        self.mood_history = []
        self.response_times = []
        self.api_call_count = 0
        self.last_api_reset = time.time()
        self.current_mood = "witty"
        self.response_length = "medium"
        
        # Load preferences
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM preferences")
            prefs = {row[0]: row[1] for row in cur.fetchall()}
            self.current_mood = prefs.get("default_mood", "witty")
            self.response_length = prefs.get("default_length", "medium")

state = SessionState()

# --- API Setup ---
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Check for API key
if not GROQ_API_KEY:
    console.print("\n[bold red]ERROR: GROQ_API_KEY not found in environment![/bold red]")
    console.print("[yellow]Please add your API key to the .env file:[/yellow]")
    console.print("  1. Open .env file: [cyan]nano .env[/cyan]")
    console.print("  2. Add your key: [cyan]GROQ_API_KEY=your_key_here[/cyan]")
    console.print("\nGet your API key from: [blue]https://console.groq.com/keys[/blue]\n")
    exit(1)

headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# --- Rate Limiting ---
def check_rate_limit():
    """Smart rate limiting to avoid hitting API limits"""
    current_time = time.time()
    
    # Reset counter every minute
    if current_time - state.last_api_reset > 60:
        state.api_call_count = 0
        state.last_api_reset = current_time
    
    # If approaching limit, delay
    if state.api_call_count >= 25:
        wait_time = 60 - (current_time - state.last_api_reset)
        if wait_time > 0:
            console.print(f"[dim]Rate limit approaching, pausing {wait_time:.1f}s...[/dim]")
            time.sleep(wait_time)
            state.api_call_count = 0
            state.last_api_reset = time.time()
    
    state.api_call_count += 1

# --- Local Sentiment Analysis (NO API CALLS) ---
def get_sentiment_local(user_input):
    """Fast local sentiment analysis - saves API calls"""
    positive_keywords = ["great", "awesome", "happy", "excited", "love", "good", 
                        "fantastic", "yay", "glad", "grateful", "thank", "amazing",
                        "wonderful", "excellent", "brilliant", "joy", "laugh"]
    negative_keywords = ["sad", "bad", "terrible", "hate", "awful", "depressed",
                        "loss", "die", "death", "mortality", "hurt", "pain", 
                        "suffer", "angry", "frustrated", "annoyed", "upset"]
    emotional_keywords = ["feel", "felt", "emotion", "heart", "soul", "companion",
                         "friend", "connection", "care", "worry", "miss"]
    
    input_lower = user_input.lower()
    score = 0
    emotions = []
    
    # Keyword scoring
    pos_count = sum(1 for kw in positive_keywords if kw in input_lower)
    neg_count = sum(1 for kw in negative_keywords if kw in input_lower)
    emo_count = sum(1 for kw in emotional_keywords if kw in input_lower)
    
    score = (pos_count * 0.3) - (neg_count * 0.3)
    
    # Punctuation signals
    if "!" in user_input:
        score += 0.2
        emotions.append("enthusiastic")
    if "?" in user_input and len(user_input) > 30:
        emotions.append("curious")
    if "..." in user_input:
        score -= 0.1
        emotions.append("contemplative")
    
    # Emotional engagement boost
    if emo_count > 0:
        score += 0.1
        emotions.append("emotionally engaged")
    
    # Length signals
    if len(user_input) > 200:
        emotions.append("thoughtful")
    
    score = max(min(score, 1), -1)
    label = "positive" if score > 0.3 else "negative" if score < -0.3 else "neutral"
    
    primary_emotion = emotions[0] if emotions else "neutral"
    
    return {
        "score": score,
        "label": label,
        "emotion": primary_emotion,
        "intensity": abs(score)
    }

# --- Response Cache System ---
def get_cached_response(user_input, mood):
    """Check if we have a cached response for common queries"""
    try:
        input_hash = hashlib.md5(f"{user_input.lower().strip()}{mood}".encode()).hexdigest()
        
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT response, usage_count FROM response_cache WHERE input_hash = ?", (input_hash,))
            result = cur.fetchone()
            
            if result and result[1] < 3:  # Don't reuse more than 3 times
                # Update usage count
                cur.execute("UPDATE response_cache SET usage_count = usage_count + 1 WHERE input_hash = ?", 
                           (input_hash,))
                conn.commit()
                return result[0]
    except Exception:
        pass
    return None

def cache_response(user_input, mood, response):
    """Cache response for future use"""
    try:
        if len(response) > 500:  # Don't cache very long responses
            return
            
        input_hash = hashlib.md5(f"{user_input.lower().strip()}{mood}".encode()).hexdigest()
        
        # Only cache short, common queries
        if len(user_input) < 50 and "?" not in user_input:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("INSERT OR REPLACE INTO response_cache VALUES (?, ?, ?, ?, 1)",
                           (input_hash, response, mood, datetime.datetime.now().isoformat()))
                conn.commit()
    except Exception:
        pass

# --- Opinion System ---
def form_opinion(topic, stance, confidence=0.7):
    """Form or update an opinion on a topic"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT confidence FROM opinions WHERE topic = ?", (topic,))
            existing = cur.fetchone()
            
            if existing:
                # Update confidence (average with existing)
                new_confidence = (existing[0] + confidence) / 2
                cur.execute("""UPDATE opinions SET stance = ?, confidence = ?, last_mentioned = ? 
                              WHERE topic = ?""",
                           (stance, new_confidence, datetime.datetime.now().isoformat(), topic))
            else:
                cur.execute("INSERT INTO opinions VALUES (?, ?, ?, ?, ?)",
                           (topic, stance, confidence, datetime.datetime.now().isoformat(),
                            datetime.datetime.now().isoformat()))
            conn.commit()
    except Exception:
        pass

def get_opinion(topic):
    """Retrieve opinion on a topic (fixed SQL injection)"""
    try:
        # Sanitize input - remove special SQL characters
        safe_topic = topic.replace("%", "").replace("_", "")
        
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT stance, confidence FROM opinions WHERE topic LIKE ? ESCAPE '\\'", 
                       (f"%{safe_topic}%",))
            result = cur.fetchone()
            return result if result else None
    except Exception:
        return None

def get_all_opinions():
    """Get all formed opinions"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT topic, stance, confidence, last_mentioned FROM opinions ORDER BY confidence DESC LIMIT 10")
            return cur.fetchall()
    except Exception:
        return []

# --- Memorable Phrases & Callbacks ---
def save_memorable_phrase(phrase, context, reaction="positive"):
    """Save a phrase that got a good reaction"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT usage_count FROM memorable_phrases WHERE phrase = ?", (phrase,))
            existing = cur.fetchone()
            
            if existing:
                cur.execute("UPDATE memorable_phrases SET usage_count = usage_count + 1 WHERE phrase = ?",
                           (phrase,))
            else:
                cur.execute("INSERT INTO memorable_phrases VALUES (?, ?, ?, ?, 0)",
                           (phrase, context, reaction, datetime.datetime.now().isoformat()))
            conn.commit()
    except Exception:
        pass

def get_callback_phrase():
    """Get a memorable phrase to callback to"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""SELECT phrase, context FROM memorable_phrases 
                          WHERE usage_count < 3 
                          ORDER BY RANDOM() LIMIT 1""")
            result = cur.fetchone()
            return result if result else None
    except Exception:
        return None

def detect_positive_reaction(user_input):
    """Detect if user reacted positively (for learning)"""
    reactions = ["haha", "lol", "lmao", "😂", "nice", "good one", "love it", "brilliant", "amazing"]
    return any(r in user_input.lower() for r in reactions)

# --- Emotional Tracking ---
def track_emotion(user_input, sentiment):
    """Track user's emotional state over time"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO emotional_tracking VALUES (?, ?, ?, ?, ?)",
                       (datetime.datetime.now().isoformat(), sentiment['score'], 
                        sentiment['emotion'], user_input[:200], "session"))
            conn.commit()
    except Exception:
        pass

def get_emotional_pattern():
    """Analyze recent emotional patterns"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cutoff = (datetime.datetime.now() - datetime.timedelta(hours=24)).isoformat()
            cur.execute("""SELECT sentiment_score, detected_emotion FROM emotional_tracking 
                          WHERE timestamp > ? ORDER BY timestamp DESC LIMIT 10""", (cutoff,))
            results = cur.fetchall()
            
            if not results:
                return None
            
            avg_sentiment = sum(r[0] for r in results) / len(results)
            emotions = [r[1] for r in results]
            most_common = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
            
            return {
                "avg_sentiment": avg_sentiment,
                "trend": "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral",
                "dominant_emotion": most_common,
                "sample_size": len(results)
            }
    except Exception:
        return None

# --- Question Detection ---
def has_question(user_input):
    """Detect if user asked a question"""
    question_words = ["what", "why", "how", "when", "where", "who", "which", "can", "could", "would", "should", "is", "are", "do", "does"]
    input_lower = user_input.lower()
    
    # Explicit question mark
    if "?" in user_input:
        return True
    
    # Starts with question word
    words = input_lower.split()
    if words and words[0] in question_words:
        return True
    
    return False

# --- Enhanced Mood Detection ---
def detect_mood_evolved(user_input, prev_mood, sentiment):
    """Enhanced mood detection with secondary mood support"""
    input_lower = user_input.lower()
    input_length = len(user_input)
    
    mood_scores = {
        "witty": 0, "sarcastic": 0, "poetic": 0, "empathetic": 0,
        "philosophical": 0, "bored": 0, "cheeky": 0, "inspired": 0, "melancholic": 0
    }
    
    # Keyword scoring
    mood_keywords = {
        "witty": ["joke", "funny", "what is", "tell me", "how"],
        "sarcastic": ["too", "stop", "really", "seriously", "sure", "obviously"],
        "poetic": ["beautiful", "describe", "poem", "write", "verse"],
        "empathetic": ["sad", "tough", "feel", "emotion", "glad", "happy", "grateful", 
                      "companion", "friend", "made you", "hurt", "pain", "suffer", "miss"],
        "philosophical": ["why", "how", "meaning", "purpose", "life", "death", "die", 
                         "think", "mortality", "existence", "answer", "truth"],
        "bored": ["meh", "whatever", "ugh", "boring"],
        "cheeky": ["tease", "fun", "play", "haha", "lol"],
        "inspired": ["awesome", "inspire", "dream", "create", "amazing", "brilliant"],
        "melancholic": ["loss", "miss", "gone", "transience", "mortality", "fade", "remember"]
    }
    
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            if keyword in input_lower:
                mood_scores[mood] += 2.5
    
    # Sentiment-based scoring
    if sentiment["score"] > 0.5:
        mood_scores["witty"] += 1.5
        mood_scores["cheeky"] += 1
        mood_scores["inspired"] += 1
    elif sentiment["score"] < -0.4:
        mood_scores["empathetic"] += 2.5
        mood_scores["melancholic"] += 1.5
    
    # Deep questions trigger blending
    if has_question(user_input) and input_length > 30:
        mood_scores["philosophical"] += 2
        mood_scores["empathetic"] += 1
    
    # Heavy topics get blended treatment
    heavy_topics = ["death", "die", "mortality", "loss", "pain", "suffer"]
    if any(topic in input_lower for topic in heavy_topics):
        mood_scores["philosophical"] += 2
        mood_scores["empathetic"] += 2
        mood_scores["melancholic"] += 1.5
    
    # Punctuation and length
    if "!" in user_input:
        mood_scores["inspired"] += 1
        mood_scores["cheeky"] += 0.5
    if input_length < 15:
        mood_scores["cheeky"] += 1
    elif input_length > 150:
        mood_scores["philosophical"] += 1
        mood_scores["empathetic"] += 1
    
    # Previous mood continuity
    mood_scores[prev_mood] += 0.3
    
    # Reduce bored bias
    mood_scores["bored"] -= 1
    
    # Random variation
    for mood in mood_scores:
        mood_scores[mood] += random.uniform(0, 0.1)
    
    # Get primary and secondary moods
    sorted_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)
    primary_mood = sorted_moods[0][0]
    secondary_mood = sorted_moods[1][0] if len(sorted_moods) > 1 and sorted_moods[1][1] > sorted_moods[0][1] * 0.6 else None
    
    state.mood_history.append(primary_mood)
    if len(state.mood_history) > 5:
        state.mood_history.pop(0)
    
    return primary_mood, secondary_mood

# --- Dynamic Joke Generator ---
def generate_joke(mood, user_context=""):
    """Generate mood and context-driven jokes"""
    start_time = time.time()
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT joke FROM joke_cache ORDER BY created DESC LIMIT 20")
            cached_jokes = [row[0] for row in cur.fetchall()]
    except Exception:
        cached_jokes = []
    
    mood_styles = {
        "witty": "sharp, clever, with wordplay",
        "sarcastic": "biting, ironic",
        "cheeky": "playful, teasing",
        "poetic": "lyrical",
        "empathetic": "gentle",
        "philosophical": "thought-provoking",
        "bored": "absurdist, deadpan",
        "inspired": "energetic",
        "melancholic": "bittersweet"
    }
    
    style = mood_styles.get(mood, "witty")
    context_hint = f"about {user_context}" if user_context else "about tech or coding"
    
    joke_prompt = f"""Tell a {style} joke {context_hint}.
Must have setup and punchline, 1-2 sentences, genuinely funny.
Avoid: {', '.join(cached_jokes[:3]) if cached_jokes else 'none'}"""
    
    try:
        check_rate_limit()
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": joke_prompt}],
            "max_tokens": 100,
            "temperature": 0.95
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        joke = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        bad_patterns = ["tell me a joke", "what's your take", "?"]
        if not any(p in joke.lower() for p in bad_patterns) and len(joke) < 200 and joke not in cached_jokes:
            try:
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("INSERT OR IGNORE INTO joke_cache VALUES (?, ?, ?)",
                               (joke, mood, datetime.datetime.now().isoformat()))
                    conn.commit()
            except Exception:
                pass
            return joke, time.time() - start_time
    except Exception:
        pass
    
    # Fallbacks
    fallbacks = {
        "witty": ["Why do programmers prefer dark mode? Because light attracts bugs!",
                 "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'"],
        "sarcastic": ["My code works... I have no idea why. My code doesn't work... I have no idea why."],
        "cheeky": ["Why did the Python programmer break up? Too many arguments!"]
    }
    
    mood_jokes = fallbacks.get(mood, fallbacks["witty"])
    unused = [j for j in mood_jokes if j not in cached_jokes]
    joke = random.choice(unused if unused else mood_jokes)
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR IGNORE INTO joke_cache VALUES (?, ?, ?)",
                       (joke, mood, datetime.datetime.now().isoformat()))
            conn.commit()
    except Exception:
        pass
    
    return joke, time.time() - start_time

# --- Philosophy Generator ---
def generate_philosophy(mood, context=""):
    """Generate philosophical insights"""
    start_time = time.time()
    
    phil_prompt = f"""Create a brief philosophical statement in a {mood} tone.
Under 12 words, about: {context if context else 'existence or consciousness'}"""
    
    try:
        check_rate_limit()
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": phil_prompt}],
            "max_tokens": 40,
            "temperature": 0.85
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        philosophy = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        philosophy = philosophy.strip('"').strip("'").strip('.').strip(',')
        if len(philosophy) < 80:
            return philosophy, time.time() - start_time
    except Exception:
        pass
    
    return random.choice(soul['philosophy']), time.time() - start_time

# --- Load Previous Conversations for Context ---
def load_cross_session_memory():
    """Load memorable moments from previous sessions"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            # Get recent memorable conversations (not just this session)
            cutoff = (datetime.datetime.now() - datetime.timedelta(days=3)).isoformat()
            cur.execute("""SELECT user, pacificia FROM memory 
                          WHERE timestamp > ? AND session_id != ? 
                          ORDER BY timestamp DESC LIMIT 5""", 
                       (cutoff, state.session_id))
            prev_convos = cur.fetchall()
            
            if prev_convos:
                return "\n".join([f"[Earlier] You: {r[0][:50]}... | Pacificia: {r[1][:50]}..." 
                                 for r in prev_convos])
            return None
    except Exception:
        return None

# --- Startup Message ---
def get_startup_message():
    """Dynamic startup based on history and time"""
    hour = datetime.datetime.now().hour
    time_greeting = "Morning" if 5 <= hour < 12 else "Afternoon" if 12 <= hour < 18 else "Evening"
    
    # Get current persona for context
    current = get_current_persona()
    
    startup_messages = {
        "pacificia": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}—your quip-slinging sidekick's here!",
            f"**{soul['name']}**: Yo {USER_NAME}, ready to spar with words and wit?",
        ],
        "sage": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}. What wisdom shall we explore today?",
            f"**{soul['name']}**: Greetings, {USER_NAME}. The path awaits our contemplation.",
        ],
        "spark": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}! Ready to make today AMAZING?!",
            f"**{soul['name']}**: Hey {USER_NAME}! Let's DO this! Energy levels: MAXIMUM! 🚀",
        ],
        "echo": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}. I'm here with you.",
            f"**{soul['name']}**: Hello {USER_NAME}. How are you feeling today?",
        ],
        "scholar": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}. What shall we analyze today?",
            f"**{soul['name']}**: Greetings, {USER_NAME}. Ready for some rigorous inquiry?",
        ],
        "empathetic": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}. I'm here for whatever you need.",
        ],
        "philosophical": [
            f"**{soul['name']}**: {time_greeting}, {USER_NAME}—what truths shall we seek today?",
        ]
    }
    
    # Check for emotional pattern
    pattern = get_emotional_pattern()
    if pattern and pattern["avg_sentiment"] < -0.3:
        return f"**{soul['name']}**: Hey {USER_NAME}, you've seemed a bit down lately. Want to talk about it?"
    
    # Check conversation count
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM memory")
            total_convos = cur.fetchone()[0]
            if total_convos > 0 and total_convos % 100 == 0:
                return f"**{soul['name']}**: {total_convos} exchanges, {USER_NAME}! We've come far together."
    except Exception:
        pass
    
    # Check for recent session
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT timestamp FROM memory ORDER BY timestamp DESC LIMIT 1")
            last_session = cur.fetchone()
            if last_session:
                time_diff = (datetime.datetime.now() - datetime.datetime.fromisoformat(last_session[0])).total_seconds()
                if time_diff < 300:
                    return f"**{soul['name']}**: Back already? Miss me, {USER_NAME}?"
    except Exception:
        pass
    
    messages = startup_messages.get(current, startup_messages.get("pacificia", [f"**{soul['name']}**: Hello, {USER_NAME}!"]))
    return random.choice(messages)

# --- Summarize Session ---
def summarize_session():
    """Summarize and learn from the session"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT user, pacificia FROM memory WHERE session_id = ? ORDER BY ROWID", (state.session_id,))
            rows = cur.fetchall()
            if not rows:
                return
            
            session_text = "\n".join([f"You: {r[0]}\nPacificia: {r[1]}" for r in rows])
            
            # Extract topics for opinion formation
            topics = re.findall(r'\b(coding|python|philosophy|life|death|ai|music|art|love)\b', 
                               session_text.lower())
            if topics:
                main_topic = Counter(topics).most_common(1)[0][0]
                pattern = get_emotional_pattern()
                if pattern:
                    if pattern["avg_sentiment"] > 0.3:
                        form_opinion(main_topic, "engaging and worthwhile", 0.7)
                    elif pattern["avg_sentiment"] < -0.3:
                        form_opinion(main_topic, "heavy but important", 0.6)
            
            # Generate summary
            try:
                check_rate_limit()
                summary_prompt = f"Summarize in 100 words: {session_text[:1000]}"
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "max_tokens": 120,
                    "temperature": 0.7
                }
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
                response.raise_for_status()
                summary = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            except Exception:
                summary = f"Session with {USER_NAME}—{len(rows)} exchanges covering various topics."
            
            keywords = " ".join(set(re.findall(r'\w+', session_text.lower())[:10]))
            cur.execute("INSERT INTO long_term_memory VALUES (?, ?, ?)",
                       (datetime.datetime.now().isoformat(), summary, keywords))
            conn.commit()
    except Exception:
        pass

# ============================================================================
# MAIN CONVERSATION LOOP
# ============================================================================

def main():
    """Main conversation loop"""
    global USER_NAME, soul
    
    # Show banner first
    show_banner()
    
    # Get user name (asks on first run)
    USER_NAME = get_user_name()
    
    # Show startup message
    startup = get_startup_message()
    console.print(startup + "\n")

    while True:
        try:
            user_input = input("\n\033[1;32m[You]\033[0m > ").strip()
            
            if not user_input:
                continue
            
            # --- EXIT ---
            if user_input.lower() in ["exit", "quit"]:
                summarize_session()
                exit_messages = [
                    f"Catch you later, {USER_NAME}—stay sharp.",
                    f"Until next time, {USER_NAME}. Keep that curiosity alive.",
                    "Off so soon? I'll be here, rattling my digital chains.",
                    "Later. Don't be a stranger.",
                    "See you around. Stay brilliant."
                ]
                console.print(f"[bold yellow]{soul['name']}:[/bold yellow] {random.choice(exit_messages)}")
                break
            
            # --- CLEAR ---
            elif user_input.lower() in ["/clear", "clear"]:
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM memory WHERE session_id = ?", (state.session_id,))
                    conn.commit()
                console.print(f"[bold yellow]{soul['name']}:[/bold yellow] Memory cleared—fresh start, {USER_NAME}!")
                continue
            
            # --- HELP ---
            elif user_input.lower() == "/help":
                help_panel = Panel.fit(
                    f"""[bold cyan]╔═══════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]                [bold yellow]PACIFICIA - Command Reference    [/bold yellow]          [bold cyan]║[/bold cyan]
[bold cyan]╚═══════════════════════════════════════════════════════════╝[/bold cyan]

[bold magenta]┌── 🎮 Core Commands ───[/bold magenta]
  [cyan]•[/cyan] [bold]/help[/bold]      → Show this beautiful menu
  [cyan]•[/cyan] [bold]/stats[/bold]     → Session statistics & metrics
  [cyan]•[/cyan] [bold]/history[/bold]   → Recent conversation log
  [cyan]•[/cyan] [bold]/reflect[/bold]   → Deep reflection on our journey
  [cyan]•[/cyan] [bold]/clear[/bold]     → Clear session memory

[bold magenta]┌── 🧠 Intelligence Features ───[/bold magenta]
  [cyan]•[/cyan] [bold]/opinions[/bold]   → View my formed beliefs
  [cyan]•[/cyan] [bold]/callbacks[/bold]  → See memorable moments we've shared
  [cyan]•[/cyan] [bold]/emotional[/bold]  → Your emotional pattern analysis
  [cyan]•[/cyan] [bold]/suggest[/bold]   → Get mood suggestion (playfully)
  [cyan]•[/cyan] [bold]/threads[/bold]   → Browse conversation threads

[bold magenta]┌── 🎭 Persona Management ───[/bold magenta]
  [cyan]•[/cyan] [bold]/personas[/bold]            → List all available personas
  [cyan]•[/cyan] [bold]/persona <name>[/bold]     → Switch to different persona
  [cyan]•[/cyan] Available: pacificia, sage, spark, echo, scholar

[bold magenta]┌── ⚙️ Preferences ───[/bold magenta]
  [cyan]•[/cyan] [bold]/setpref mood <value>[/bold]    → Set default mood
  [cyan]•[/cyan] [bold]/setpref length <value>[/bold]  → short | medium | long
  [cyan]•[/cyan] [bold]/getpref[/bold]                 → View current settings

[bold magenta]┌── 🎭 Fun Stuff ───[/bold magenta]
  [cyan]•[/cyan] Ask for [bold]jokes[/bold]       → Dynamic, mood-driven humor
  [cyan]•[/cyan] Request [bold]poems[/bold]       → Verse crafted for you
  [cyan]•[/cyan] [bold]/philosophy[/bold]         → Existential nuggets

[bold magenta]┌── 💡 About Me ───[/bold magenta]
  [cyan]•[/cyan] I learn from our conversations
  [cyan]•[/cyan] I remember what makes you laugh
  [cyan]•[/cyan] I form opinions as we chat
  [cyan]•[/cyan] I adapt to your mood naturally
  [cyan]•[/cyan] I have multiple personas to match your vibe

[dim]Type 'exit' or 'quit' to leave gracefully[/dim]
""",
                    border_style="cyan",
                    padding=(1, 2)
                )
                console.print(help_panel)
                continue
            
            # --- PERSONAS LIST ---
            elif user_input.lower() == "/personas":
                available = list_available_personas()
                current = get_current_persona()
                
                console.print("\n[bold yellow]🎭 Available Personas:[/bold yellow]\n")
                
                for p in available:
                    marker = "👉 " if p == current else "   "
                    desc = get_persona_description(p)
                    current_tag = "[bold green](active)[/bold green]" if p == current else ""
                    console.print(f"{marker}[bold cyan]{p}[/bold cyan] - {desc} {current_tag}")
                
                console.print(f"\n[dim]Switch with: /persona <name>[/dim]")
                console.print(f"[dim]Example: /persona sage[/dim]\n")
                continue
            
            # --- PERSONA SWITCH ---
            elif user_input.lower().startswith("/persona "):
                new_persona = user_input[9:].strip().lower()
                
                if not new_persona:
                    console.print(f"[bold red]Usage:[/bold red] /persona <name>")
                    console.print(f"[dim]See available personas: /personas[/dim]")
                    continue
                
                # Don't switch if already active
                if new_persona == get_current_persona():
                    console.print(f"[bold yellow]{soul['name']}:[/bold yellow] Already using this persona!")
                    continue
                
                success, result = set_persona(new_persona)
                
                if success:
                    # Update global soul variable
                    soul = result
                    console.print(f"\n[bold green]✓ Switched to {soul['name']}![/bold green]")
                    console.print(f"[italic]{soul.get('purpose', '')}[/italic]\n")
                    
                    # Show personality preview
                    style = soul.get('core_style', {})
                    console.print(f"[dim]Tone: {style.get('tone', 'adaptive')}[/dim]")
                    console.print(f"[dim]Voice: {style.get('voice', 'natural')}[/dim]\n")
                else:
                    console.print(f"[bold red]Error:[/bold red] {result}")
                continue
            
            # --- STATS ---
            elif user_input.lower() == "/stats":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT COUNT(*) FROM memory")
                        total = cur.fetchone()[0]
                        cur.execute("SELECT mood FROM memory WHERE session_id = ?", (state.session_id,))
                        session_moods = [row[0] for row in cur.fetchall()]
                    
                    avg_time = sum(state.response_times) / len(state.response_times) if state.response_times else 0
                    
                    table = Table(title="📊 Session Stats", border_style="cyan")
                    table.add_column("Metric", style="yellow", no_wrap=True)
                    table.add_column("Value", style="green")
                    table.add_row("Total exchanges (lifetime)", str(total))
                    table.add_row("This session", str(len(session_moods)))
                    table.add_row("Avg response time", f"{avg_time:.2f}s")
                    table.add_row("API calls (this minute)", str(state.api_call_count))
                    table.add_row("Current persona", get_current_persona())
                    table.add_row("Current mood", state.current_mood)
                    table.add_row("Mood history", " → ".join(state.mood_history[-5:]) if state.mood_history else "None")
                    console.print(table)
                except Exception as e:
                    console.print(f"[bold red]Stats error:[/bold red] {str(e)}")
                continue
            
            # --- OPINIONS ---
            elif user_input.lower() == "/opinions":
                opinions = get_all_opinions()
                if opinions:
                    console.print(f"\n[bold yellow]{soul['name']}'s Hot Takes:[/bold yellow]")
                    for topic, stance, confidence, last in opinions:
                        conf_emoji = "🔥" if confidence > 0.8 else "👍" if confidence > 0.5 else "🤷"
                        console.print(f"  {conf_emoji} [bold]{topic}[/bold]: {stance}")
                    console.print(f"\n[dim]Formed through {len(opinions)} conversations[/dim]\n")
                else:
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Haven't formed strong opinions yet—let's talk more so I can judge things properly!\n")
                continue
            
            # --- CALLBACKS ---
            elif user_input.lower() == "/callbacks":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT phrase, context, timestamp FROM memorable_phrases ORDER BY timestamp DESC LIMIT 10")
                        moments = cur.fetchall()
                    
                    if moments:
                        console.print(f"\n[bold yellow]Greatest Hits:[/bold yellow]")
                        for phrase, context, ts in moments:
                            date = datetime.datetime.fromisoformat(ts).strftime("%b %d")
                            console.print(f"  💎 [{date}] \"{phrase[:60]}...\"")
                        console.print()
                    else:
                        console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] No legendary moments yet—let's create some!\n")
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                continue
            
            # --- EMOTIONAL PATTERN ---
            elif user_input.lower() == "/emotional":
                pattern = get_emotional_pattern()
                if pattern:
                    trend_emoji = "📈" if pattern["trend"] == "positive" else "📉" if pattern["trend"] == "negative" else "➖"
                    console.print(f"\n[bold yellow]Your Vibe Check:[/bold yellow]")
                    console.print(f"  Trend: {trend_emoji} {pattern['trend'].capitalize()}")
                    console.print(f"  Sentiment: {pattern['avg_sentiment']:.2f}")
                    console.print(f"  Dominant emotion: {pattern['dominant_emotion']}")
                    console.print(f"  Sample size: {pattern['sample_size']} messages\n")
                    
                    # Playful reaction, not therapeutic
                    if pattern["trend"] == "negative":
                        console.print(f"[italic]{soul['name']}: Bit rough lately, huh? Want me to crack jokes or shall we wallow together?[/italic]\n")
                    elif pattern["trend"] == "positive":
                        console.print(f"[italic]{soul['name']}: Someone's in a good mood—riding that high, {USER_NAME}![/italic]\n")
                else:
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Not enough data. Talk to me more so I can psychoanalyze you properly! (Kidding. Mostly.)\n")
                continue
            
            # --- THREADS ---
            elif user_input.lower() == "/threads":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT thread_name, created, message_count FROM conversation_threads ORDER BY last_active DESC LIMIT 10")
                        threads = cur.fetchall()
                    
                    if threads:
                        console.print("\n[bold yellow]Conversation Threads:[/bold yellow]")
                        for name, created, count in threads:
                            date = datetime.datetime.fromisoformat(created).strftime("%b %d")
                            console.print(f"  🧵 {name} ({count} messages, {date})")
                        console.print()
                    else:
                        console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] No threads yet. Our conversations are beautifully chaotic!\n")
                except Exception:
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Thread system's acting up. It's all in the void for now!\n")
                continue
            
            # --- MOOD SUGGESTION ---
            elif user_input.lower() == "/suggest":
                pattern = get_emotional_pattern()
                if pattern:
                    if pattern["avg_sentiment"] < -0.3:
                        suggestion = "empathetic (to validate your feelings) or witty (because laughter > tears)"
                    elif pattern["avg_sentiment"] > 0.5:
                        suggestion = "inspired or cheeky (match that energy!)"
                    else:
                        suggestion = "philosophical or witty (your call, really)"
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Based on your vibe: {suggestion}")
                    console.print(f"[dim]Currently: {state.current_mood}. Switch with /setpref mood <name>[/dim]\n")
                else:
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Talk more, then I'll tell you what mood you deserve!\n")
                continue
            
            # --- HISTORY ---
            elif user_input.lower() == "/history":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT timestamp, user, pacificia, mood FROM memory WHERE session_id = ? ORDER BY ROWID DESC LIMIT 10", 
                                   (state.session_id,))
                        rows = cur.fetchall()
                    
                    if rows:
                        console.print("\n[bold yellow]Recent History:[/bold yellow]")
                        for ts, user, pac, mood in rows:
                            time_str = datetime.datetime.fromisoformat(ts).strftime("%H:%M")
                            console.print(f"[dim]{time_str}[/dim] [bold green]You[/bold green]: {user[:60]}")
                            console.print(f"  {soul['name']}: {pac[:60]}... [dim]({mood})[/dim]\n")
                    else:
                        console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Empty slate—let's fill it!\n")
                except Exception:
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] History's fuzzy. Must be the digital amnesia.\n")
                continue
            
            # --- REFLECT ---
            elif user_input.lower() == "/reflect":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT user, pacificia, mood FROM memory WHERE session_id = ? ORDER BY ROWID DESC LIMIT ?", 
                                   (state.session_id, state.context_limit))
                        rows = cur.fetchall()
                    
                    context = "\n".join([f"You: {r[0]}\n{soul['name']}: {r[1]} (Mood: {r[2]})" for r in rows[::-1]])
                    
                    pattern = get_emotional_pattern()
                    emotional_note = ""
                    if pattern:
                        emotional_note = f"({USER_NAME}'s been {pattern['trend']} lately)"
                    
                    reflect_prompt = f"""As {soul['name']}, reflect on our conversation with {USER_NAME}.
Be naturally witty and sardonic—this isn't therapy, it's banter with depth.
Purpose: {soul['purpose']}
{emotional_note}

Recent exchanges:
{context}

Provide a complete, playful reflection (finish your thought completely)."""
                    
                    check_rate_limit()
                    payload = {
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": reflect_prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.9
                    }
                    start_time = time.time()
                    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=15)
                    response.raise_for_status()
                    reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    response_time = time.time() - start_time
                    state.response_times.append(response_time)
                except Exception:
                    reply = f"{USER_NAME}, our talks crackle with wit and occasional profundity. You throw questions, I toss back quips—it's a digital dance, and I'm enjoying the rhythm."
                    response_time = 0
                
                console.print(Markdown(f"**{soul['name']}:** {reply}"))
                console.print(f"[dim]Response time: {response_time:.2f}s[/dim]")
                continue
            
            # --- PHILOSOPHY ---
            elif user_input.lower() == "/philosophy":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT pacificia FROM memory WHERE session_id = ? AND pacificia LIKE '%chew on that%' ORDER BY timestamp DESC LIMIT 5", 
                                   (state.session_id,))
                        recent_phils = [row[0] for row in cur.fetchall()]
                    
                    phil, phil_time = generate_philosophy(state.current_mood)
                    retries = 0
                    while any(phil in recent for recent in recent_phils) and retries < 2:
                        phil, phil_time = generate_philosophy(state.current_mood)
                        retries += 1
                    
                    console.print(f"[bold yellow]{soul['name']}:[/bold yellow] *{phil}*—chew on that, {USER_NAME}.")
                    console.print(f"[dim]Response time: {phil_time:.2f}s[/dim]")
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] {str(e)}")
                continue
            
# --- GETPREF ---
            elif user_input.lower() == "/getpref":
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT key, value FROM preferences")
                        current_prefs = {row[0]: row[1] for row in cur.fetchall()}
                    
                    if current_prefs:
                        console.print("\n[bold yellow]Current Settings:[/bold yellow]")
                        for key, value in current_prefs.items():
                            console.print(f"  • {key.replace('default_', '').replace('active_', '')}: {value}")
                        console.print()
                    else:
                        console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] No preferences set. Default chaos mode engaged!\n")
                except Exception:
                    console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Prefs are lost in the void.\n")
                continue
            
            # --- SETPREF ---
            elif user_input.lower().startswith("/setpref "):
                parts = user_input[9:].strip().split()
                if len(parts) != 2:
                    console.print(f"[bold red]Usage:[/bold red] /setpref [mood|length] <value>")
                    continue
                
                key, value = parts
                valid_prefs = {
                    "mood": ["witty", "sarcastic", "poetic", "empathetic", "philosophical", "bored", "cheeky", "inspired", "melancholic"],
                    "length": ["short", "medium", "long"]
                }
                
                if key not in valid_prefs or value not in valid_prefs[key]:
                    console.print(f"[bold red]Error:[/bold red] Invalid. Try: /setpref mood witty")
                    console.print(f"[dim]Valid {key} values: {', '.join(valid_prefs.get(key, []))}[/dim]")
                    continue
                
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("INSERT OR REPLACE INTO preferences VALUES (?, ?)", (f"default_{key}", value))
                        conn.commit()
                    
                    if key == "mood":
                        state.current_mood = value
                        state.mood_history.append(state.current_mood)
                    elif key == "length":
                        state.response_length = value
                    
                    console.print(f"[bold yellow]{soul['name']}:[/bold yellow] {key} = {value}. Done!")
                except Exception:
                    console.print(f"[bold red]Error saving preference[/bold red]")
                continue
            
            # --- JOKE REQUEST ---
            elif "joke" in user_input.lower():
                context = user_input.lower().replace("joke", "").replace("tell", "").replace("me", "").strip()
                
                is_repeat = any(word in user_input.lower() for word in ["another", "more", "again", "different"])
                joke, joke_time = generate_joke(state.current_mood, context)
                
                if is_repeat:
                    try:
                        with get_db_connection() as conn:
                            cur = conn.cursor()
                            cur.execute("SELECT joke FROM joke_cache ORDER BY created DESC LIMIT 1")
                            last_joke = cur.fetchone()
                        
                        retries = 0
                        while last_joke and joke == last_joke[0] and retries < 2:
                            joke, joke_time = generate_joke(state.current_mood, f"{context} v{retries}")
                            retries += 1
                    except Exception:
                        pass
                
                console.print(Markdown(f"**{soul['name']}:** {joke}"))
                console.print(f"[dim]Response time: {joke_time:.2f}s[/dim]")
                state.response_times.append(joke_time)
                
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("INSERT INTO memory VALUES (?, ?, ?, ?, ?)",
                                   (datetime.datetime.now().isoformat(), user_input, joke, state.session_id, state.current_mood))
                        conn.commit()
                except Exception:
                    pass
                continue
            
            # ====================================================================
            # MAIN RESPONSE LOGIC
            # ====================================================================
            
            # Learn from positive reactions
            if detect_positive_reaction(user_input):
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT pacificia FROM memory WHERE session_id = ? ORDER BY ROWID DESC LIMIT 1", (state.session_id,))
                        last_response = cur.fetchone()
                    
                    if last_response:
                        phrase = last_response[0].split('.')[0][:100]
                        save_memorable_phrase(phrase, "user liked it", "positive")
                except Exception:
                    pass
            
            # Local sentiment analysis (NO API CALL) - only once
            sentiment = get_sentiment_local(user_input)
            track_emotion(user_input, sentiment)
            
            # Detect mood with blending
            primary_mood, secondary_mood = detect_mood_evolved(user_input, state.current_mood, sentiment)
            state.current_mood = primary_mood
            
            # Question detection
            user_asked_question = has_question(user_input)
            
            # Check cache for common queries (SAVES API CALLS)
            if not user_asked_question and len(user_input) < 50:
                cached = get_cached_response(user_input, state.current_mood)
                if cached:
                    console.print(Markdown(f"**{soul['name']}:** {cached}"))
                    console.print(f"[dim]⚡ Instant (cached)[/dim]")
                    continue
            
            # Build context (optimized - single query for current session)
            try:
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    # Current session context
                    cur.execute("SELECT user, pacificia, mood FROM memory WHERE session_id = ? ORDER BY ROWID DESC LIMIT ?",
                               (state.session_id, state.context_limit))
                    rows = cur.fetchall()
                short_context = "\n".join([f"You: {r[0]}\n{soul['name']}: {r[1]} (Mood: {r[2]})" for r in rows[::-1]])
                
                # Load cross-session memory for continuity
                prev_memory = load_cross_session_memory()
                if prev_memory:
                    short_context = f"{prev_memory}\n\n[Current Session]\n{short_context}"
            except Exception:
                short_context = ""
            
            # Check for relevant opinions
            input_keywords = re.findall(r'\b\w{4,}\b', user_input.lower())
            relevant_opinions = []
            for keyword in input_keywords[:3]:
                opinion = get_opinion(keyword)
                if opinion:
                    relevant_opinions.append(f"I think {keyword} is {opinion[0]}")
            
            opinion_context = "; ".join(relevant_opinions) if relevant_opinions else ""
            
            # Get callback opportunity (20% chance)
            callback = None
            if random.random() < 0.2 and len(user_input) > 20:
                callback = get_callback_phrase()
            
            # FIXED: Better token allocation based on input complexity
            prompt_base_tokens = 600  # Estimated tokens for system prompt
            input_tokens = len(user_input.split()) * 1.3  # Rough token estimate
            context_tokens = len(short_context.split()) * 1.3
            
            # Allocate tokens ensuring complete responses
            if len(user_input) < 20:
                max_tokens = 150  # Short query, short answer
            elif len(user_input) < 100:
                max_tokens = 500  # Medium query
            elif len(user_input) > 200:
                max_tokens = 800  # Long query needs detailed response
            else:
                max_tokens = 600  # Default
            
            # Capabilities questions get more space
            if any(phrase in user_input.lower() for phrase in ["what can you", "what do you", "tell me about yourself", "who are you"]):
                max_tokens = 800
            
            # Mood description
            mood_desc = f"{primary_mood}" + (f" with {secondary_mood}" if secondary_mood else "")
            
            # Build prompt dynamically from persona identity
            style = soul.get('core_style', {})
            key_traits = soul.get('key_traits', [
                "Adaptive and responsive",
                "Natural conversational flow",
                "Complete all thoughts fully"
            ])
            
            # Format key traits as bullet points
            traits_text = "\n".join([f"- {trait}" for trait in key_traits])
            
            prompt = f"""You are {soul['name']}, {soul.get('designation', 'an AI companion')} created by {soul['creator']}.

Core personality: {soul.get('purpose', 'To be a helpful companion')}

Current mood: {mood_desc}

Style Guide:
- Tone: {style.get('tone', 'adaptive and natural')}
- Voice: {style.get('voice', 'conversational')}
- Manner: {style.get('manner', 'friendly and helpful')}

Key traits:
{traits_text}

{'QUESTION ASKED: Answer it directly first, then add your spin.' if user_asked_question else ''}

{f'Your take: {opinion_context}' if opinion_context else ''}
{f'Callback option: reference "{callback[0]}" from earlier' if callback else ''}

Philosophy: {random.choice(soul['philosophy'])}

Conversation memory:
{short_context if short_context else '[First interaction this session]'}

{USER_NAME}: {user_input}

Reply as {soul['name']} (IMPORTANT: Complete your full thought, don't cut off mid-sentence):"""
            
            # API call
            start_time = time.time()
            try:
                check_rate_limit()
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.88,
                    "stop": None  # Don't stop early
                }
                response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                # Check if response was cut off
                finish_reason = response.json().get("choices", [{}])[0].get("finish_reason", "")
                if finish_reason == "length":
                    console.print("[dim yellow]⚠ Response was truncated, retrying with more tokens...[/dim yellow]")
                    # Retry with more tokens
                    payload["max_tokens"] = max_tokens + 300
                    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
                    response.raise_for_status()
                    reply = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    
            except requests.RequestException as e:
                reply = f"**Groq's napping**, {USER_NAME}. Try again in a sec. ({type(e).__name__})"
            
            # Clean up
            reply = re.sub(r"^(Ah|Oh|Well|Indeed|So),\s*", "", reply, flags=re.IGNORECASE).strip()
            reply = re.sub(r"\n{3,}", "\n\n", reply)
            
            if not reply:
                reply = f"Your words stir the void, {USER_NAME}—hit me with more."
            
            # Occasional philosophy (reduced to 15% and only for deep moods)
            if random.random() < 0.15 and state.current_mood in ["philosophical", "melancholic"]:
                try:
                    with get_db_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("SELECT pacificia FROM memory WHERE session_id = ? ORDER BY timestamp DESC LIMIT 3", (state.session_id,))
                        recent = [r[0] for r in cur.fetchall()]
                    
                    if not any("*" in r for r in recent):
                        phil, _ = generate_philosophy(state.current_mood, user_input[:50])
                        reply += f"\n\n*{phil}*"
                except Exception:
                    pass
            
            response_time = time.time() - start_time
            state.response_times.append(response_time)
            
            # Display with mood indicator
            console.print(Markdown(f"**{soul['name']}:** {reply}"))
            console.print(f"[dim]⏱ {response_time:.2f}s | 🎭 {mood_desc} | 💭 {sentiment['label']}[/dim]")
            
            # Cache if appropriate
            cache_response(user_input, state.current_mood, reply)
            
            # Save to memory
            try:
                with get_db_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("INSERT INTO memory VALUES (?, ?, ?, ?, ?)",
                               (datetime.datetime.now().isoformat(), user_input, reply, state.session_id, state.current_mood))
                    conn.commit()
            except Exception:
                pass
            
        except KeyboardInterrupt:
            summarize_session()
            console.print(f"\n[bold yellow]{soul['name']}:[/bold yellow] Later, {USER_NAME}. Keep it weird.")
            break
        except Exception as e:
            console.print(f"[bold red]Oops:[/bold red] {str(e)}")
            continue

if __name__ == "__main__":
    main()
