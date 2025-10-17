"""
Migration script to add model_hash column to generated_images table
"""
import sqlite3
import os

# Database paths
DB_PATHS = [
    "sd_webui.db",
    "backend/sd_webui.db"
]

def add_model_hash_column():
    """Add missing columns to generated_images table"""
    # Columns to add (column_name, column_type)
    columns_to_add = [
        ("image_hash", "TEXT"),
        ("source_image_hash", "TEXT"),
        ("mask_data", "TEXT"),
        ("lora_names", "TEXT"),
        ("model_hash", "TEXT"),
    ]

    for db_path in DB_PATHS:
        if not os.path.exists(db_path):
            print(f"Database not found: {db_path}")
            continue

        print(f"\nProcessing database: {db_path}")

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check existing columns
            cursor.execute("PRAGMA table_info(generated_images)")
            existing_columns = [row[1] for row in cursor.fetchall()]

            # Add missing columns
            for column_name, column_type in columns_to_add:
                if column_name in existing_columns:
                    print(f"  [SKIP] Column '{column_name}' already exists")
                else:
                    cursor.execute(f"""
                        ALTER TABLE generated_images
                        ADD COLUMN {column_name} {column_type}
                    """)
                    print(f"  [OK] Added column '{column_name}'")

            conn.commit()
            conn.close()
            print(f"  [DONE] Completed processing {db_path}")

        except Exception as e:
            print(f"  [ERROR] Error processing {db_path}: {e}")
            if conn:
                conn.close()

if __name__ == "__main__":
    print("Adding model_hash column to generated_images table...")
    add_model_hash_column()
    print("\nMigration completed!")
