# Raw Data

Place raw EMA data files in this directory.

## Expected Format

Raw data should be in CSV format with the following structure:
- `worker_id`: Unique identifier for each worker
- `timestamp`: Timestamp for each observation
- Feature columns: Various EMA measurements

## Example

```csv
worker_id,timestamp,mood_score,stress_level,sleep_hours
001,2024-01-01 08:00:00,7.5,3.2,7.0
001,2024-01-01 12:00:00,6.8,4.1,7.0
002,2024-01-01 08:00:00,8.2,2.5,8.5
...
```
