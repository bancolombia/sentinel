# Ingestion

## Why this module exists

Enterprise systems generate logs in wildly different formats — free-text lines,
key-value pairs, nested XML fragments, concatenated JSON blobs. Before any
analysis can happen, these raw files must be converted into structured tabular
data (rows and columns). The Ingestion module solves this first-mile problem
so you can focus on analysis instead of parsing.

## When to use it

- You have raw `.log` or `.txt` files from WAS, HSM, HDC, IBMMQ, or ZTNA systems.
- You want a one-liner to go from file path → pandas DataFrame.
- You need to build a custom parser for a proprietary log format.

## Built-in Parsers

| Parser | Log Format | Key Output Columns |
|--------|-----------|-------------------|
| `WASParser` | WebSphere Application Server | `timestamp`, `log_level`, `service`, `transaction_id`, `amount`, `response_code` |
| `HSMParser` | Hardware Security Module | `date`, `time`, `level`, `ip`, `category`, `message` |
| `HDCParser` | High-Density Computing | `timestamp`, `thread_id`, `log_source`, `log_type`, `message`, `error_code` |
| `IBMMQParser` | IBM Message Queue | `Process`, `Program`, `Host`, `Time`, `CodigoAMQ_Error`, `Explanation`, `Action` |
| `ZTNAParser` | Cloudflare Zero Trust | `Action`, `Datetime`, `SourceIP`, `DestinationIP`, `Email`, `PolicyName` |

## Using LogIngestor (dispatch)

The simplest approach — `LogIngestor` picks the right parser for you:

```python
from sentinel.ingestion import LogIngestor

df = LogIngestor.ingest("path/to/logfile.log", log_type="WAS")
print(df.shape)       # (n_entries, n_columns)
print(df.dtypes)      # all columns start as object/string
print(df.head())
```

Supported type strings: `"HSM"`, `"HDC"`, `"IBM_MQ"`, `"WAS"`, `"ZTNA"` (case-insensitive).

### What to expect in the output

The returned DataFrame has **one row per log entry** and **one column per
extracted field**. All values are initially strings — you will typically
need to:

1. Convert timestamps: `df["timestamp"] = pd.to_datetime(df["timestamp"])`
2. Convert numerics: `df["amount"] = pd.to_numeric(df["amount"], errors="coerce")`
3. Set the index: `df = df.set_index("timestamp").sort_index()`

If the file is empty, unreadable, or contains no matching patterns, the
parser returns an **empty DataFrame** rather than raising an exception.
Always check `df.empty` before proceeding.

## Using a parser directly

```python
from sentinel.ingestion import WASParser

parser = WASParser("path/to/was_log.log")
df = parser.parse()
```

This is equivalent to `LogIngestor.ingest(..., log_type="WAS")` but gives
you direct access to the parser instance.

## Creating a custom parser

Subclass `BaseLogParser` and implement the `parse()` method. The constructor
receives `file_path` as its only argument.

```python
from sentinel.ingestion import BaseLogParser
import pandas as pd

class MyAppParser(BaseLogParser):
    """Parser for my custom application logs.

    Expected format: TIMESTAMP|LEVEL|MESSAGE
    """

    def parse(self) -> pd.DataFrame:
        records = []
        with open(self.file_path) as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) == 3:
                    records.append({
                        "timestamp": parts[0],
                        "level": parts[1],
                        "message": parts[2],
                    })
        return pd.DataFrame(records)

parser = MyAppParser("path/to/custom.log")
df = parser.parse()
```

### Design guidelines for custom parsers

- **Return an empty DataFrame on failure** — don't raise exceptions for
  malformed lines. Skip them and log a warning if needed.
- **Extract as many fields as possible** — it's easier to drop columns
  later than to re-parse the file.
- **Keep timestamps as strings** — let the downstream consumer decide
  the format and timezone.
- **Handle encoding** — enterprise logs often use `latin-1` or `utf-8`
  with replacement characters. Use `errors="replace"` when opening files.

## Interpreting the output

After ingestion, inspect your DataFrame to understand what you're working with:

```python
# How many log entries were parsed?
print(f"Entries: {len(df)}")

# What columns are available?
print(df.columns.tolist())

# How many null values per column?
print(df.isnull().sum())

# What's the time range?
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"From: {df['timestamp'].min()}")
print(f"To:   {df['timestamp'].max()}")
print(f"Span: {df['timestamp'].max() - df['timestamp'].min()}")
```

A healthy ingestion result typically shows:
- **Thousands to millions of rows** (enterprise logs are verbose)
- **Low null rates** in key columns (timestamp, level)
- **Higher null rates** in optional fields (transaction_id, amount) — this is normal
- **Consistent time range** matching the expected log period

If you see very few rows or high null rates in critical columns, the parser
may not match your log format. Consider writing a custom parser.

## What comes next

Once you have a structured DataFrame, the typical next steps are:

1. **Transformer** — aggregate raw entries into time-windowed features
2. **Explorer** — validate whether the data contains meaningful signals
3. **Detectors** — run anomaly detection on the prepared features
