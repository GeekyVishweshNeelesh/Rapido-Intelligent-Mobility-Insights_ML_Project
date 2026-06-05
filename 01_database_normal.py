"""
RAPIDO ML PROJECT - NORMALIZED DATABASE SETUP
================================================
Creates database with 5 normalized tables following PDF specification
Uses PyMySQL connector (compatible with MariaDB)

Tables:
- bookings
- customers
- drivers
- location_demand
- time_features

Author: VishweshN
Database: MariaDB
"""

import pymysql
from pymysql import Error
import pandas as pd
from sqlalchemy import create_engine, text
import os

# ============================================
# DATABASE CONFIGURATION
# ============================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'vishwesh',
    'password': 'Vish1408',    # ⚠️ CHANGE THIS!
    'database': 'rapido_ml_normalized',
    'charset': 'utf8mb4'
}

# Path to your 5 CSV files
CSV_PATH = '/home/vishwesh/Documents/GUVI_Course_Projects/Rapido_Project/data/'  # ⚠️ UPDATE THIS!

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
    """Create the normalized Rapido ML database"""

    print("="*70)
    print("🗄️  DATABASE SETUP: NORMALIZED (5 TABLES)")
    print("="*70)
    print("\nSTEP 1: CREATING DATABASE")
    print("-"*70)

    connection = get_connection(include_db=False)
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # Drop if exists and recreate
        print(f"\n⚠️  Dropping '{DB_CONFIG['database']}' if it exists...")
        cursor.execute(f"DROP DATABASE IF EXISTS {DB_CONFIG['database']}")

        print(f"✅ Creating database '{DB_CONFIG['database']}'...")
        cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")

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
# FUNCTION 2: CREATE NORMALIZED TABLES
# ============================================

def create_tables():
    """Create 5 normalized tables"""

    print("\n" + "="*70)
    print("STEP 2: CREATING 5 NORMALIZED TABLES")
    print("-"*70)

    connection = get_connection(include_db=True)
    if not connection:
        return False

    try:
        cursor = connection.cursor()

        # ─── TABLE 1: BOOKINGS ───
        print("\n📊 Creating 'bookings' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                booking_id      VARCHAR(100),
                customer_id     VARCHAR(100),
                driver_id       VARCHAR(100),
                vehicle_type    VARCHAR(50),
                pickup_location VARCHAR(150),
                drop_location   VARCHAR(150),
                pickup_city     VARCHAR(100),
                drop_city       VARCHAR(100),
                booking_date    DATE,
                booking_time    VARCHAR(50),
                ride_distance   FLOAT,
                ride_duration   FLOAT,
                base_fare       FLOAT,
                surge_multiplier FLOAT,
                total_fare      FLOAT,
                payment_method  VARCHAR(50),
                booking_status  VARCHAR(50),
                cancellation_reason TEXT,
                cancelled_by    VARCHAR(50),
                weather_condition VARCHAR(50),
                traffic_level   VARCHAR(50),
                INDEX idx_booking_id    (booking_id),
                INDEX idx_customer      (customer_id),
                INDEX idx_driver        (driver_id),
                INDEX idx_status        (booking_status),
                INDEX idx_vehicle       (vehicle_type),
                INDEX idx_pickup_city   (pickup_city)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("   ✅ 'bookings' created")

        # ─── TABLE 2: CUSTOMERS ───
        print("\n👥 Creating 'customers' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                customer_id         VARCHAR(100),
                total_bookings      INT,
                completed_rides     INT,
                cancelled_rides     INT,
                cancellation_rate   FLOAT,
                average_rating      FLOAT,
                preferred_payment   VARCHAR(50),
                preferred_vehicle   VARCHAR(50),
                peak_hour_bookings  INT,
                loyalty_tier        VARCHAR(50),
                signup_date         DATE,
                INDEX idx_customer_id       (customer_id),
                INDEX idx_cancellation_rate (cancellation_rate),
                INDEX idx_rating            (average_rating),
                INDEX idx_loyalty           (loyalty_tier)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("   ✅ 'customers' created")

        # ─── TABLE 3: DRIVERS ───
        print("\n🚗 Creating 'drivers' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id                  INT AUTO_INCREMENT PRIMARY KEY,
                driver_id           VARCHAR(100),
                total_rides         INT,
                completed_rides     INT,
                cancelled_rides     INT,
                total_delays        INT,
                delay_rate          FLOAT,
                average_rating      FLOAT,
                acceptance_rate     FLOAT,
                vehicle_type        VARCHAR(50),
                experience_years    FLOAT,
                reliability_score   FLOAT,
                performance_tier    VARCHAR(50),
                INDEX idx_driver_id     (driver_id),
                INDEX idx_delay_rate    (delay_rate),
                INDEX idx_rating        (average_rating),
                INDEX idx_reliability   (reliability_score),
                INDEX idx_vehicle       (vehicle_type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("   ✅ 'drivers' created")

        # ─── TABLE 4: LOCATION DEMAND ───
        print("\n📍 Creating 'location_demand' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS location_demand (
                id                      INT AUTO_INCREMENT PRIMARY KEY,
                location_name           VARCHAR(150),
                city                    VARCHAR(100),
                avg_demand              FLOAT,
                peak_hour_demand        FLOAT,
                avg_surge_multiplier    FLOAT,
                cancellation_rate       FLOAT,
                avg_ride_distance       FLOAT,
                avg_fare                FLOAT,
                INDEX idx_location  (location_name),
                INDEX idx_city      (city),
                INDEX idx_demand    (avg_demand)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("   ✅ 'location_demand' created")

        # ─── TABLE 5: TIME FEATURES ───
        print("\n⏰ Creating 'time_features' table...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_features (
                id              INT AUTO_INCREMENT PRIMARY KEY,
                booking_id      VARCHAR(100),
                hour            INT,
                day_of_week     INT,
                day_name        VARCHAR(20),
                month           INT,
                month_name      VARCHAR(20),
                quarter         INT,
                year            INT,
                is_weekend      TINYINT(1),
                is_peak_hour    TINYINT(1),
                is_rush_hour    TINYINT(1),
                is_night        TINYINT(1),
                season          VARCHAR(20),
                INDEX idx_booking_id    (booking_id),
                INDEX idx_hour          (hour),
                INDEX idx_day           (day_of_week),
                INDEX idx_month         (month)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        print("   ✅ 'time_features' created")

        connection.commit()
        print("\n" + "="*70)
        print("✅ ALL 5 TABLES CREATED SUCCESSFULLY!")
        print("="*70)
        return True

    except Error as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        connection.close()

# ============================================
# FUNCTION 3: LOAD CSV DATA
# ============================================

def load_csv_to_mariadb():
    """Load 5 CSV files into normalized tables"""

    print("\n" + "="*70)
    print("STEP 3: LOADING 5 CSV FILES")
    print("-"*70)

    # SQLAlchemy engine using pymysql
    engine = create_engine(
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?charset={DB_CONFIG['charset']}"
    )

    # CSV to Table mapping
    csv_files = {
        'bookings.csv'       : 'bookings',
        'customers.csv'      : 'customers',
        'drivers.csv'        : 'drivers',
        'location_demand.csv': 'location_demand',
        'time_features.csv'  : 'time_features'
    }

    loaded_count = 0

    try:
        for csv_file, table_name in csv_files.items():
            csv_path = os.path.join(CSV_PATH, csv_file)

            print(f"\n📂 {csv_file} → '{table_name}'")

            if not os.path.exists(csv_path):
                print(f"   ⚠️  File not found: {csv_path}")
                print(f"   Skipping...")
                continue

            # Read CSV
            df = pd.read_csv(csv_path)
            print(f"   Rows: {len(df):,}  |  Columns: {len(df.columns)}")

            # Convert datetime columns
            for col in df.columns:
                if 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                    except:
                        pass

            # Handle NaN and inf
            df = df.fillna('')

            # Load to database (replace to avoid duplicates)
            df.to_sql(
                name=table_name,
                con=engine,
                if_exists='replace',
                index=False,
                chunksize=1000
            )

            print(f"   ✅ Loaded successfully!")
            loaded_count += 1

        print("\n" + "="*70)
        print(f"✅ LOADED {loaded_count}/5 CSV FILES SUCCESSFULLY!")
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
# FUNCTION 4: VERIFY DATA
# ============================================

def verify_data():
    """Verify all tables have data"""

    print("\n" + "="*70)
    print("STEP 4: VERIFYING DATA")
    print("-"*70)

    connection = get_connection(include_db=True)
    if not connection:
        return False

    try:
        cursor = connection.cursor()
        tables = ['bookings', 'customers', 'drivers', 'location_demand', 'time_features']

        print(f"\n{'Table':<25} {'Rows':>12} {'Columns':>12}")
        print("-"*50)

        total_rows = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cursor.fetchone()[0]
            cursor.execute(f"SHOW COLUMNS FROM {table}")
            col_count = len(cursor.fetchall())
            print(f"{table:<25} {row_count:>12,} {col_count:>12}")
            total_rows += row_count

        print("-"*50)
        print(f"{'TOTAL':<25} {total_rows:>12,}")

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
    print("🗄️  RAPIDO ML - NORMALIZED DATABASE SETUP")
    print("="*70)

    if not create_database(): exit(1)
    if not create_tables():   exit(1)
    if not load_csv_to_mariadb(): exit(1)
    verify_data()

    print("\n" + "="*70)
    print("🎉 NORMALIZED DATABASE SETUP COMPLETE!")
    print("="*70)
    print(f"\n✅ Database : {DB_CONFIG['database']}")
    print(f"✅ Tables   : 5 normalized tables")
    print(f"✅ Ready for Streamlit deployment!")
    print("="*70)
