import time
import os
import sqlite3
import requests
import logging
from dotenv import load_dotenv

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_name="financials.db"):
        self.db_name = db_name
        self._init_db()

    def _init_db(self):
        """Creates the database and tables if they don't exist."""
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarterly_financials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    fiscal_year INTEGER,
                    fiscal_period TEXT,
                    end_date TEXT,
                    filing_date TEXT,
                    revenues REAL,
                    gross_profit REAL,
                    operating_income REAL,
                    net_income REAL,
                    eps_basic REAL,
                    eps_diluted REAL,
                    total_assets REAL,
                    total_liabilities REAL,
                    total_equity REAL,
                    operating_cash_flow REAL,
                    UNIQUE(symbol, fiscal_year, fiscal_period)
                )
            """)
            conn.commit()

    def insert_financials(self, records):
        """Inserts a list of financial records into the database."""
        if not records:
            return

        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            # INSERT OR IGNORE silently skips records that already exist (based on the UNIQUE constraint)
            cursor.executemany(
                """
                INSERT OR IGNORE INTO quarterly_financials 
                (symbol, fiscal_year, fiscal_period, end_date, filing_date, 
                 revenues, gross_profit, operating_income, net_income,
                 eps_basic, eps_diluted, total_assets, total_liabilities, total_equity, operating_cash_flow)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                records,
            )
            conn.commit()
            logger.info(f"Inserted/Updated {len(records)} records for {records[0][0]}.")


class BulkPolygonScraper:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY must be set in the .env file.")

        self.base_url = "https://api.polygon.io"
        self.last_request_time = 0.0
        self.db = DatabaseManager()

    def _rate_limit_wait(self):
        """Enforces Polygon's 5 requests/minute free tier limit (1 request every 12.5s)."""
        elapsed = time.time() - self.last_request_time
        if elapsed < 13.0:
            time.sleep(13.0 - elapsed)
        self.last_request_time = time.time()

    def fetch_all_history(self, symbol: str):
        """
        Fetches up to 100 quarters (25 years) of historical financials in a single API call.
        """
        self._rate_limit_wait()

        url = f"{self.base_url}/vX/reference/financials?ticker={symbol.upper()}&timeframe=quarterly&limit=100&apiKey={self.api_key}"

        try:
            response = requests.get(url)
            if response.status_code == 429:
                logger.warning(f"[{symbol}] Rate limited! Sleeping for 60 seconds...")
                time.sleep(60)
                return self.fetch_all_history(symbol)  # Retry

            response.raise_for_status()
            results = response.json().get("results", [])

            parsed_records = []
            for rep in results:
                # Extract identifiers
                fy = rep.get("fiscal_year")
                fp = rep.get("fiscal_period")
                end_date = rep.get("end_date")
                filing_date = rep.get("filing_date")

                # Grab the separate statements
                financials_obj = rep.get("financials", {})
                income_stmt = financials_obj.get("income_statement", {})
                balance_sheet = financials_obj.get("balance_sheet", {})
                cash_flow = financials_obj.get("cash_flow_statement", {})

                # Helper to grab values safely since Polygon nests them
                def get_val(statement, keys):
                    for k in keys:
                        if k in statement and "value" in statement[k]:
                            return float(statement[k]["value"])
                    return None

                # 1. Income Statement
                rev = get_val(income_stmt, ["revenues", "operating_revenue"])
                gp = get_val(income_stmt, ["gross_profit"])
                oi = get_val(income_stmt, ["operating_income_loss"])
                ni = get_val(
                    income_stmt,
                    [
                        "net_income_loss",
                        "net_income_loss_available_to_common_stockholders_basic",
                    ],
                )
                eps_basic = get_val(income_stmt, ["basic_earnings_per_share"])
                eps_diluted = get_val(income_stmt, ["diluted_earnings_per_share"])

                # 2. Balance Sheet
                assets = get_val(balance_sheet, ["assets"])
                liabilities = get_val(balance_sheet, ["liabilities"])
                equity = get_val(
                    balance_sheet,
                    [
                        "equity",
                        "stockholders_equity",
                        "stockholders_equity_including_portion_attributable_to_noncontrolling_interest",
                    ],
                )

                # 3. Cash Flow Statement
                ocf = get_val(
                    cash_flow,
                    [
                        "net_cash_flow_from_operating_activities",
                        "net_cash_flow_from_operating_activities_continuing",
                    ],
                )

                # Append as a tuple matching the SQLite columns
                if fy and fp:
                    parsed_records.append(
                        (
                            symbol,
                            fy,
                            fp,
                            end_date,
                            filing_date,
                            rev,
                            gp,
                            oi,
                            ni,
                            eps_basic,
                            eps_diluted,
                            assets,
                            liabilities,
                            equity,
                            ocf,
                        )
                    )

            # Save to database
            self.db.insert_financials(parsed_records)

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")

    def run(self, symbols: list):
        logger.info(f"Starting bulk scrape for {len(symbols)} symbols...")
        for i, symbol in enumerate(symbols):
            logger.info(f"Processing {symbol} ({i+1}/{len(symbols)})...")
            self.fetch_all_history(symbol)
        logger.info(f"Bulk scrape complete! Data saved to {self.db.db_name}")


if __name__ == "__main__":
    logger.info("Loading custom symbols...")
    try:
        with open('symbols.txt', 'r') as f:
            raw_symbols = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        logger.error("symbols.txt not found.")
        raw_symbols = []

    TARGET_SYMBOLS = [symbol.replace("-", ".") for symbol in raw_symbols]

    scraper = BulkPolygonScraper()
    scraper.run(TARGET_SYMBOLS)
