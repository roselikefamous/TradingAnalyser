import sqlite3
import pandas as pd
import datetime
import os

DB_PATH = "terminal_data.db"

def get_connection():
    # Check if we are running in a different directory
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Watchlist Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            date_added TEXT NOT NULL
        )
    ''')
    
    # 2. Journal Table (Only one row needed essentially, or historical)
    c.execute('''
        CREATE TABLE IF NOT EXISTS journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            last_updated TEXT NOT NULL
        )
    ''')
    
    # 3. Portfolio / Simulation Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            sl REAL NOT NULL,
            tp REAL NOT NULL,
            shares REAL NOT NULL,
            open_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN',
            close_price REAL,
            close_date TEXT,
            pnl REAL
        )
    ''')
    
    # 4. Strategies Table (AI Trading Engine)
    c.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            source_book TEXT,
            logic_json TEXT NOT NULL,
            base_weight REAL DEFAULT 1.0
        )
    ''')
    
    # 5. Trade Feedback Table (ML Loop)
    c.execute('''
        CREATE TABLE IF NOT EXISTS trade_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER,
            strategy_id INTEGER,
            rating INTEGER,
            comment TEXT,
            FOREIGN KEY(strategy_id) REFERENCES strategies(id)
        )
    ''')
    
    # Pre-fill watchlist if entirely empty (first launch)
    c.execute("SELECT COUNT(*) FROM watchlist")
    if c.fetchone()[0] == 0:
        default_wl = ["AAPL", "MSFT", "GOOG", "TSLA", "BTC-USD"]
        now = datetime.datetime.now().isoformat()
        c.executemany("INSERT INTO watchlist (symbol, date_added) VALUES (?, ?)",
                      [(sym, now) for sym in default_wl])
                      
    # Pre-fill journal if entirely empty
    c.execute("SELECT COUNT(*) FROM journal")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO journal (content, last_updated) VALUES (?, ?)",
                  ("", datetime.datetime.now().isoformat()))
                  
    # Pre-fill default strategies if empty
    c.execute("SELECT COUNT(*) FROM strategies")
    if c.fetchone()[0] == 0:
        _prefill_default_strategies(c)
                  
    conn.commit()
    conn.close()

def _prefill_default_strategies(cursor):
    import json
    # Dummy mock strategies translated from "books"
    strat1 = {
        "description": "Buy when RSI is oversold and price crosses above EMA.",
        "conditions": [
            {"indicator": "RSI_14", "operator": "<", "value": 35},
            {"indicator": "Close", "operator": ">", "value": "EMA_21"}
        ],
        "action": "BUY"
    }
    
    strat2 = {
        "description": "Trend Following Breakout",
        "conditions": [
            {"indicator": "EMA_9", "operator": ">", "value": "EMA_21"},
            {"indicator": "ADX_14", "operator": ">", "value": 25}
        ],
        "action": "BUY"
    }

    cursor.execute("INSERT INTO strategies (name, source_book, logic_json, base_weight) VALUES (?, ?, ?, ?)",
                   ("RSI Reversal", "Mastering the Trade (John Carter)", json.dumps(strat1), 1.0))
    cursor.execute("INSERT INTO strategies (name, source_book, logic_json, base_weight) VALUES (?, ?, ?, ?)",
                   ("Moving Average Trend", "Trend Following (Michael Covel)", json.dumps(strat2), 1.0))


# --- WATCHLIST CRUD ---
def get_watchlist():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT symbol FROM watchlist ORDER BY id ASC")
    wl = [row[0] for row in c.fetchall()]
    conn.close()
    return wl

def add_to_watchlist(symbol: str):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO watchlist (symbol, date_added) VALUES (?, ?)", 
                  (symbol.upper(), datetime.datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass # Already exists
    conn.close()

def remove_from_watchlist(symbol: str):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
    conn.close()

# --- JOURNAL CRUD ---
def get_journal():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT content FROM journal ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()
    return row[0] if row else ""

def save_journal(content: str):
    conn = get_connection()
    c = conn.cursor()
    # Update the only row (assuming id=1 exists from init_db)
    # Alternatively, keep history by inserting new rows. We'll just update.
    c.execute("UPDATE journal SET content = ?, last_updated = ? WHERE id = (SELECT MAX(id) FROM journal)",
              (content, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

# --- PORTFOLIO CRUD (used by simulation state mapping) ---
def get_open_positions():
    """Returns a list of dicts matching the structure of sim_state['positions']"""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM portfolio WHERE status = 'OPEN'", conn)
    conn.close()
    
    positions = []
    for _, row in df.iterrows():
        positions.append({
            'db_id': row['id'],           # Hidden tracking ID
            'symbol': row['symbol'],
            'direction': row['direction'],
            'entry_price': row['entry_price'],
            'sl': row['sl'],
            'tp': row['tp'],
            'shares': row['shares'],
            'open_date': row['open_date']
        })
    return positions

def get_closed_positions():
    """Returns a list of dicts matching the structure of sim_state['trade_history']"""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM portfolio WHERE status = 'CLOSED'", conn)
    conn.close()
    
    history = []
    for _, row in df.iterrows():
        history.append({
            'symbol': row['symbol'],
            'direction': row['direction'],
            'entry_price': row['entry_price'],
            'close_price': row['close_price'],
            'shares': row['shares'],
            'pnl': row['pnl'],
            'open_date': row['open_date'],
            'close_date': row['close_date']
        })
    return history

def add_position(pos: dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO portfolio (symbol, direction, entry_price, sl, tp, shares, open_date, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
    ''', (pos['symbol'], pos.get('direction', 'BUY'), pos['entry_price'], pos['sl'], pos['tp'], pos['shares'], pos['open_date']))
    pos_id = c.lastrowid
    conn.commit()
    conn.close()
    return pos_id

def close_position(db_id: int, close_price: float, close_date: str, pnl: float):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        UPDATE portfolio 
        SET status = 'CLOSED', close_price = ?, close_date = ?, pnl = ?
        WHERE id = ?
    ''', (close_price, close_date, pnl, db_id))
    conn.commit()
    conn.close()

def reset_portfolio():
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM portfolio")
    conn.commit()
    conn.close()

# --- AI STRATEGY ENGINE CRUD ---
def get_all_strategies():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, source_book, logic_json, base_weight FROM strategies ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    
    strats = []
    for r in rows:
        strats.append({
            "id": r[0],
            "name": r[1],
            "source_book": r[2],
            "logic_json": r[3],
            "base_weight": r[4]
        })
    return strats

def add_strategy(name: str, source_book: str, logic_json: str, base_weight: float = 1.0):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO strategies (name, source_book, logic_json, base_weight)
        VALUES (?, ?, ?, ?)
    ''', (name, source_book, logic_json, base_weight))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    return sid

def update_strategy_weight(strategy_id: int, new_weight: float):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE strategies SET base_weight = ? WHERE id = ?", (new_weight, strategy_id))
    conn.commit()
    conn.close()

def add_trade_feedback(trade_id: int, strategy_id: int, rating: int, comment: str = ""):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO trade_feedback (trade_id, strategy_id, rating, comment)
        VALUES (?, ?, ?, ?)
    ''', (trade_id, strategy_id, rating, comment))
    conn.commit()
    conn.close()
