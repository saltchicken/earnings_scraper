import os
import copy
import time
import json
import requests
from dotenv import load_dotenv

# Fallback in case algotrader.logger isn't available in the execution environment
try:
    from algotrader.logger import get_logger

    logger = get_logger(__name__)
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)


class PolygonClient:
    def __init__(self):
        """
        Initializes the Polygon API client.
        Requires POLYGON_API_KEY to be set in the .env file.
        """
        load_dotenv()
        self.api_key = os.getenv("POLYGON_API_KEY")

        if not self.api_key:
            raise ValueError(
                "POLYGON_API_KEY must be set in the .env file to use Polygon data."
            )

        self.base_url = "https://api.polygon.io"

    def _make_request(self, url: str, max_retries: int = 2) -> requests.Response:
        """
        Internal helper to make requests with automatic retry on 429 (Rate Limit).
        Polygon's free tier limits users to 5 requests per minute.
        """
        wait_time = 60

        for attempt in range(max_retries):
            response = requests.get(url)

            if response.status_code == 429:
                logger.warning(
                    f"Polygon rate limit hit. Retrying in {wait_time}s "
                    f"(Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                return response

        # Return the last response if all retries are exhausted
        return response

    def get_ticker_details(self, symbol: str) -> dict:
        """
        Fetches general company details from Polygon (market cap, employees, etc).
        """
        url = f"{self.base_url}/v3/reference/tickers/{symbol.upper()}?apiKey={self.api_key}"
        response = self._make_request(url)

        if response.status_code == 200:
            return response.json().get("results", {})

        logger.error(
            f"Failed to fetch Polygon ticker details for {symbol}: {response.text}"
        )
        return {}

    def get_historical_financials(self, symbol: str, limit: int = 1) -> list:
        """
        Fetches point-in-time historical quarterly financials.
        Automatically detects missing quarters (e.g., dropped Q4s) and forward-fills
        them using the previous quarter's data to maintain strict ML time-series alignment.
        """
        # Fetch extra records internally (limit * 2) so we have historical buffer data to fill gaps
        url = f"{self.base_url}/vX/reference/financials?ticker={symbol.upper()}&timeframe=quarterly&limit={limit * 2}&apiKey={self.api_key}"
        response = self._make_request(url)

        if response.status_code != 200:
            logger.error(
                f"Failed to fetch Polygon financials for {symbol}: {response.text}"
            )
            return []

        raw_results = response.json().get("results", [])
        if not raw_results:
            return []

        continuous_results = []

        # Iterate from Newest to Oldest to construct a perfect timeline
        for i in range(len(raw_results)):
            continuous_results.append(raw_results[i])

            # Stop if we've successfully gathered the exact amount the user requested
            if len(continuous_results) >= limit:
                break

            # Compare current (newer) report with the next (older) report in the list
            if i + 1 < len(raw_results):
                curr_rep = raw_results[i]
                next_rep = raw_results[i + 1]

                curr_q_str = curr_rep.get("fiscal_period", "")
                curr_y = curr_rep.get("fiscal_year")
                next_q_str = next_rep.get("fiscal_period", "")
                next_y = next_rep.get("fiscal_year")

                # Ensure we have valid quarterly string formats before doing math
                if not (
                    curr_q_str.startswith("Q")
                    and next_q_str.startswith("Q")
                    and curr_y
                    and next_y
                ):
                    continue

                curr_q = int(curr_q_str.replace("Q", ""))

                # Calculate what the next OLDER quarter logically should be
                expected_older_q = curr_q - 1
                expected_older_y = int(curr_y)

                if expected_older_q == 0:
                    expected_older_q = 4
                    expected_older_y -= 1

                next_q = int(next_q_str.replace("Q", ""))

                # Loop to forward-fill gaps (e.g., jump from 2025 Q1 to 2024 Q3 implies missing 2024 Q4)
                while (
                    expected_older_q != next_q or expected_older_y != int(next_y)
                ) and len(continuous_results) < limit:
                    logger.warning(
                        f"Missing SEC filing detected for {symbol}: {expected_older_y} Q{expected_older_q}. "
                        f"Forward-filling from {int(next_y)} Q{next_q}."
                    )

                    # Deep copy the older available quarter and carry its values FORWARD into the missing gap
                    imputed_rep = copy.deepcopy(next_rep)
                    imputed_rep["fiscal_period"] = f"Q{expected_older_q}"
                    imputed_rep["fiscal_year"] = expected_older_y

                    continuous_results.append(imputed_rep)

                    # Decrement expectation again in case multiple quarters in a row are missing
                    expected_older_q -= 1
                    if expected_older_q == 0:
                        expected_older_q = 4
                        expected_older_y -= 1

        return continuous_results[:limit]


class EarningsScanner:
    def __init__(
        self,
        symbols: list,
        threshold_pct: float = 10.0,
        state_file: str = "scanner_state.json",
    ):
        """
        Initializes the Earnings Scanner.
        :param symbols: List of ticker symbols to track.
        :param threshold_pct: Minimum positive percentage increase to trigger a notification.
        :param state_file: Path to JSON file to track the last seen filing dates.
        """
        self.symbols = symbols
        self.threshold_pct = threshold_pct
        self.state_file = state_file
        self.client = PolygonClient()
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Loads the last seen filing dates from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state file: {e}. Starting fresh.")
        return {}

    def _save_state(self):
        """Saves the current filing dates to disk."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving state file: {e}")

    def _extract_net_income(self, report: dict):
        """
        Helper method to traverse Polygon's nested financials response
        to safely extract the net income value.
        """
        try:
            income_stmt = report.get("financials", {}).get("income_statement", {})

            # Polygon uses varying keys depending on how the company reports it
            possible_keys = [
                "net_income_loss",
                "net_income_loss_available_to_common_stockholders_basic",
                "comprehensive_income_loss",
            ]

            for key in possible_keys:
                if key in income_stmt and "value" in income_stmt[key]:
                    return float(income_stmt[key]["value"])

        except Exception as e:
            logger.error(f"Error parsing net income structure: {e}")

        return None

    def _send_notification(
        self,
        symbol: str,
        pct_change: float,
        curr_ni: float,
        prev_ni: float,
        filing_date: str,
    ):
        """
        Handles pushing out the alert. Replace/expand this with Slack, Email, or SMS logic.
        """
        msg = (
            f"🚨 EARNINGS ALERT: {symbol} 🚨\n"
            f"Filing Date: {filing_date}\n"
            f"Net Income surged by {pct_change:.2f}%!\n"
            f"Previous Qtr: ${prev_ni:,.2f} | Current Qtr: ${curr_ni:,.2f}"
        )
        logger.info(f"\n{'-'*40}\n{msg}\n{'-'*40}")

        # Example Slack webhook hookout:
        # requests.post("YOUR_SLACK_WEBHOOK_URL", json={"text": msg})

    def run(self):
        """Executes the daily scan loop for all configured symbols."""
        logger.info(f"Starting earnings scan for {len(self.symbols)} symbols...")

        for symbol in self.symbols:
            logger.info(f"Checking {symbol}...")

            # Fetch the latest 2 quarters (index 0 = Current/Newest, index 1 = Previous/Older)
            financials = self.client.get_historical_financials(symbol, limit=2)

            if len(financials) < 2:
                logger.warning(
                    f"[{symbol}] Not enough historical data to compare quarters. Skipping."
                )
                continue

            curr_report = financials[0]
            prev_report = financials[1]

            curr_filing_date = curr_report.get("filing_date")
            last_seen_date = self.state.get(symbol)

            # 1. Initialize state if symbol is completely new
            if not last_seen_date:
                logger.info(
                    f"[{symbol}] First time scanning. Initializing state with filing date {curr_filing_date}."
                )
                self.state[symbol] = curr_filing_date
                self._save_state()
                continue

            # 2. Check if a new report has dropped
            if curr_filing_date and curr_filing_date > last_seen_date:
                logger.info(
                    f"[{symbol}] New report detected! Filed on {curr_filing_date}."
                )

                curr_ni = self._extract_net_income(curr_report)
                prev_ni = self._extract_net_income(prev_report)

                if curr_ni is not None and prev_ni is not None:
                    # Calculate percentage change
                    if prev_ni == 0:
                        pct_change = float("inf") if curr_ni > 0 else float("-inf")
                    else:
                        pct_change = ((curr_ni - prev_ni) / abs(prev_ni)) * 100

                    logger.info(f"[{symbol}] Net Income Change: {pct_change:.2f}%")

                    # 3. Trigger Notification if threshold met
                    if pct_change >= self.threshold_pct:
                        self._send_notification(
                            symbol, pct_change, curr_ni, prev_ni, curr_filing_date
                        )
                else:
                    logger.warning(
                        f"[{symbol}] Could not extract net income from the financials payload."
                    )

                # 4. Update state to reflect we've processed this new report
                self.state[symbol] = curr_filing_date
                self._save_state()
            else:
                logger.debug(f"[{symbol}] No new reports since {last_seen_date}.")

        logger.info("Scan complete.")


if __name__ == "__main__":
    # Define the list of symbols you want to track
    TARGET_SYMBOLS = ["AAPL", "MSFT", "NVDA", "TSLA"]

    # Notify if net income is up 15% or more QoQ
    scanner = EarningsScanner(symbols=TARGET_SYMBOLS, threshold_pct=15.0)

    # Run the scanner (hook this up to a daily cron job)
    scanner.run()
