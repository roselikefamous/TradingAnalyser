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
                  
    conn.commit()
    conn.close()

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

