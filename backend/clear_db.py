import sqlite3

conn = sqlite3.connect('app.db')
conn.execute('PRAGMA foreign_keys = OFF')

# Show counts before deletion
u = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
t = conn.execute('SELECT COUNT(*) FROM tokens').fetchone()[0]
s = conn.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
print(f'Before: users={u}, tokens={t}, scans={s}')

# Delete all data (order matters due to FK)
conn.execute('DELETE FROM scans')
conn.execute('DELETE FROM tokens')
conn.execute('DELETE FROM users')

# Reset autoincrement counters so IDs start from 1 again
conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('users','tokens','scans')")

conn.commit()

# Verify
u2 = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
t2 = conn.execute('SELECT COUNT(*) FROM tokens').fetchone()[0]
s2 = conn.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
print(f'After:  users={u2}, tokens={t2}, scans={s2}')
print('Done! All accounts and scan history deleted. IDs reset to 1.')
conn.close()
