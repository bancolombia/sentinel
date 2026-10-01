"""Example: creating a custom log parser by extending BaseLogParser."""

import os
import tempfile

import pandas as pd
from sentinel.ingestion import BaseLogParser


class SimpleCSVLogParser(BaseLogParser):
    """Parser for simple CSV-formatted log files."""

    def parse(self) -> pd.DataFrame:
        """Read a CSV log file into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Parsed log data.
        """
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Error reading {self.file_path}: {e}")
            return pd.DataFrame()


def main():
    # Create a sample CSV log file in the OS temp directory so the example works
    # cross-platform (Windows, macOS, Linux).
    sample_data = "timestamp,level,message\n2025-01-01 00:00:00,INFO,Service started\n2025-01-01 00:01:00,ERROR,Connection timeout\n"
    sample_path = os.path.join(tempfile.gettempdir(), "sample_log.csv")

    with open(sample_path, "w") as f:
        f.write(sample_data)

    # Parse it
    parser = SimpleCSVLogParser(sample_path)
    df = parser.parse()
    print(df)
    print(f"\nParsed {len(df)} log entries")


if __name__ == "__main__":
    main()
