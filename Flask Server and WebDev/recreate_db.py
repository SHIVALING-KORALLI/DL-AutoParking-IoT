# recreate_db.py
from app import db, User
from datetime import datetime

def recreate_database():
    # Drop existing tables
    db.drop_all()
    
    # Create new tables with updated schema
    db.create_all()
    
    print("Database recreated successfully!")

if __name__ == "__main__":
    recreate_database()