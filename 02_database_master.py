"""
RAPIDO ML PROJECT - MASTER DATABASE SETUP
================================================
Creates database with 1 master table (engineered dataset)
Uses PyMySQL connector (compatible with MariaDB)

Table:
- master_dataset (115+ features, all targets)

Author: VishweshN
Database: MariaDB
"""

import pymysql
from pymysql import Error
import pandas as pd
from sqlalchemy import create_engine
import os

# ============================================
# DATABASE CONFIGURATION
# ============================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'vishwesh',
    'password': 'Vish1408',   # ⚠️ CHANGE THIS!
    'database': 'rapido_ml_master',
    'charset': 'utf8mb4'
}

# Path to your master engineered CSV
CSV_PATH = r'/home/vishwesh/Documents/GUVI_Course_Projects/Rapido_Project/'  # ⚠️ UPDATE THIS!
CSV_FILE = 'master_dataset_engineered.csv'

# ============================================
# HELPER FUNCTION: GET CONNECTION
# ============================================

def get_connection(include_db=True):
    """Get PyMySQL connection to MariaDB"""
    try:
        config = {
            'host': DB_CONFIG['host'],
            'port': DB_CONFIG['port'],
            'user': DB_CONFIG['user'],
            'password': DB_CONFIG['password'],
            'charset': DB_CONFIG['charset']
        }
        if include_db:
            config['database'] = DB_CONFIG['database']

        connection = pymysql.connect(**config)
        return connection
    except Error as e:
        print(f"❌ Connection failed: {e}")
        return None

# ============================================
# FUNCTION 1: CREATE DATABASE
# ============================================

def create_database():
    """Create the master Rapido ML database"""

    print("="*70)
    print("🗄️  DATABASE SETUP: MASTER TABLE (ALL FEATURES)")
    print("="*70)
    print("\nSTEP 1: CREATING DATABASE")
    print("-"*70)

    connection = get_connection(include_db=False)
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        print(f"\n⚠️  Dropping '{DB_CONFIG['database']}' if it exists...")
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG['database']}")

        print(f"✅ Creating database '{DB_CONFIG['database']}'...")
        cursor.execute(
            f"CREATE DATABASE {DB_CONFIG['database']} "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )

        connection.commit()
        print(f"✅ Database '{DB_CONFIG['database']}' created successfully!")
        return True

    except Error as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

# ============================================
# FUNCTION 2: LOAD MASTER DATASET
# ============================================

def load_master_dataset():
    """Load master engineered dataset into MariaDB"""

    print("\n" + "="*70)
    print("STEP 2: LOADING MASTER ENGINEERED DATASET")
    print("-"*70)

    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?charset={DB_CONFIG['charset']}"
    )

    try:
        csv_path = os.path.join(CSV_PATH, CSV_FILE)

        # Check if file exists
        if not os.path.exists(csv_path):
            print(f"\n❌ File not found: {csv_path}")
            print(f"\n💡 TO FIX - Run this in Colab Notebook 2:")
            print(f"   import os")
            print(f"   os.makedirs('{CSV_PATH}', exist_ok=True)")
            print(f"   df.to_csv('{csv_path}', index=False)")
            return False

        print(f"\n📂 File found: {CSV_FILE}")

        # Read CSV
        print(f"📊 Reading CSV...")
        df = pd.read_csv(csv_path)

        print(f"\n📈 Dataset Statistics:")
        print(f"   Rows     : {len(df):,}")
        print(f"   Columns  : {len(df.columns):,}")
        print(f"   Size     : {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

        # Show target variables found
        target_cols = [col for col in df.columns if 'target' in col.lower()]
        if target_cols:
            print(f"\n🎯 Target Variables Found ({len(target_cols)}):")
            for col in target_cols:
                print(f"   • {col}")

        # Convert datetime columns
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols:
            print(f"\n⏰ Processing {len(datetime_cols)} datetime columns...")
            for col in datetime_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass

        # Handle NaN and Inf
        print(f"\n🔧 Cleaning NaN and Inf values...")
        df = df.replace([float('inf'), float('-inf')], 0)
        df = df.fillna(0)

        # Load to MariaDB
        print(f"\n💾 Loading to MariaDB 'master_dataset' table...")
        print(f"   Please wait - loading {len(df):,} rows...")

        df.to_sql(
            name='master_dataset',
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=5000,
            method='multi'
        )

        print(f"\n✅ Successfully loaded {len(df):,} rows!")

        # Create indexes
        print(f"\n🔧 Creating indexes...")

        connection = get_connection(include_db=True)
        cursor = connection.cursor()

        index_columns = [
            'booking_id', 'customer_id', 'driver_id',
            'booking_status', 'vehicle_type',
            'pickup_city', 'drop_city'
        ]

        indexes_created = 0
        for col in index_columns:
            if col in df.columns:
                try:
                    cursor.execute(f"CREATE INDEX idx_{col} ON master_dataset(`{col}`)")
                    print(f"   ✅ Index on '{col}'")
                    indexes_created += 1
                except:
                    pass

        connection.commit()
        cursor.close()
        connection.close()

        print(f"\n   Total indexes created: {indexes_created}")
        print("\n" + "="*70)
        print("✅ MASTER DATASET LOADED SUCCESSFULLY!")
        print("="*70)
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        engine.dispose()

# ============================================
# FUNCTION 3: VERIFY DATA
# ============================================

def verify_data():
    """Verify that data was loaded correctly"""

    print("\n" + "="*70)
    print("STEP 3: VERIFYING DATA")
    print("-"*70)

    connection = get_connection(include_db=True)
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Row count
        cursor.execute("SELECT COUNT(*) FROM master_dataset")
        row_count = cursor.fetchone()[0]

        # Column count
        cursor.execute("SHOW COLUMNS FROM master_dataset")
        columns = cursor.fetchall()
        col_count = len(columns)

        # Table size
        cursor.execute(f"""
            SELECT ROUND(((data_length + index_length) / 1024 / 1024), 2)
            FROM information_schema.TABLES
            WHERE table_schema = '{DB_CONFIG['database']}'
            AND table_name = 'master_dataset'
        """)
        size_result = cursor.fetchone()
        size_mb = size_result[0] if size_result else 'N/A'

        print(f"\n📊 DATABASE STATISTICS:")
        print("-"*50)
        print(f"   Table    : master_dataset")
        print(f"   Rows     : {row_count:,}")
        print(f"   Columns  : {col_count:,}")
        print(f"   Size(MB) : {size_mb}")
        print("-"*50)

        # Show target columns
        target_cols = [col[0] for col in columns if 'target' in col[0].lower()]
        if target_cols:
            print(f"\n🎯 Target Variables:")
            for t in target_cols:
                print(f"   • {t}")

        print("\n✅ Verification complete!")
        return True

    except Error as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("🗄️  RAPIDO ML - MASTER DATABASE SETUP")
    print("="*70)

    if not create_database():        exit(1)
    if not load_master_dataset():    exit(1)
    verify_data()

    print("\n" + "="*70)
    print("🎉 MASTER DATABASE SETUP COMPLETE!")
    print("="*70)
    print(f"\n✅ Database : {DB_CONFIG['database']}")
    print(f"✅ Table    : master_dataset")
    print(f"✅ Features : 115+ engineered features")
    print(f"✅ Ready for Streamlit deployment!")
    print("="*70)
