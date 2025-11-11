import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

LOG_PATH = Path(__file__).resolve().parent / "../outputs/inference/pipeline_speed.csv"
HEADER = [
    "timestamp",
    "stage",
    "sim_id",
    "infer_id",
    "obs_id",
    "num_sims",
    "num_obs",
    "details",
    "seconds",
    "status",
]

def _ensure_logfile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(HEADER)

def append_runtime_log(
    *,
    stage: str,
    seconds: float,
    sim_id: Optional[str] = None,
    infer_id: Optional[str] = None,
    obs_id: Optional[str] = None,
    num_sims: Optional[str] = None,
    num_obs: Optional[str] = None,
    details: str = "",
    status: str = "success",
    log_path: Path = LOG_PATH,
) -> None:
    """Append a runtime measurement row to the shared CSV log."""
    _ensure_logfile(log_path)
    ts = datetime.now().isoformat(timespec="seconds")
    row = [
        ts,
        stage,
        sim_id or "",
        infer_id or "",
        obs_id or "",
        str(num_sims or ""),
        str(num_obs or ""),
        details,
        f"{float(seconds):.6f}",
        status,
    ]
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow(row)

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Append a runtime entry to the pipeline log.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seconds", required=True, type=float)
    parser.add_argument("--sim_id")
    parser.add_argument("--infer_id")
    parser.add_argument("--obs_id")
    parser.add_argument("--num_sims")
    parser.add_argument("--num_obs")
    parser.add_argument("--details", default="")
    parser.add_argument("--status", default="success")
    parser.add_argument("--log_path", default=str(LOG_PATH))
    args = parser.parse_args()
    append_runtime_log(
        stage=args.stage,
        seconds=args.seconds,
        sim_id=args.sim_id,
        infer_id=args.infer_id,
        obs_id=args.obs_id,
        num_sims=args.num_sims,
        num_obs=args.num_obs,
        details=args.details,
        status=args.status,
        log_path=Path(args.log_path).expanduser().resolve(),
    )

if __name__ == "__main__":
    _cli()
