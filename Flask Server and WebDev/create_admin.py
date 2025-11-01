# create_admin.py
from app import app, db, User
from werkzeug.security import generate_password_hash

with app.app_context():
    admin = User(
        name="Admin",
        phone_number="1234567890",
        number_plate="ADMIN001",
        password=generate_password_hash("adminpassword"),
        role="management"
    )
    db.session.add(admin)
    db.session.commit()
    print("✅ Management user created successfully!")
