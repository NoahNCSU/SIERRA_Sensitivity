import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple
import glob
import copy
import time

# ============================================================
# USER CONFIG
# ============================================================
ATD_PATH = Path(r"C:\SIERRA\Models\ATD\ATD.exe")
CASE_ROOT_PATH = Path(r"C:\Users\naetter\OneDrive - North Carolina State University\Desktop\EAB Optimization Project")

# Base input JSON to use for the control case (Design Basis Accidents)
BASE_INPUT_JSON = Path(
    r"C:\Users\naetter\OneDrive - North Carolina State University\Desktop\EAB Optimization Project\Design_Basis_Accidents\Design_Basis_Accidents_Elevated\inputs\input.json"
)

# Provide either a list of MET files or a glob pattern.
# If MET_FILES is non-empty, it is used and MET_GLOB is ignored.
MET_FILES: List[str] = [
    r"C:/SIERRA/Test_Cases_ATD/MET_8387.nrc",
]
MET_GLOB = r"C:/SIERRA/Test_Cases_ATD/*.nrc"

# Run the control case first for each MET file
RUN_CONTROL = True

# Sensitivity configuration (user-adjustable bounds/deltas)
# For numeric parameters: provide min/max/delta
# For categorical parameters: provide values list
SENSITIVITY_CONFIG: Dict[str, Dict[str, Any]] = {
    "stack_height": {"path": ["source_info", "stack_height"], "min": 10.0, "max": 80.0, "delta": 5.0},
    "stack_dia": {"path": ["source_info", "stack_dia"], "min": 0.05, "max": 2.0, "delta": 0.05},
    "stack_flow": {"path": ["source_info", "stack_flow"], "min": 0.0, "max": 5.0, "delta": 0.5},
    "stack_terrain": {"path": ["source_info", "stack_terrain"], "min": 0.0, "max": 50.0, "delta": 5.0},
    "stack_heat_emis": {"path": ["source_info", "stack_heat_emis"], "min": 0.0, "max": 5.0, "delta": 0.5},
    "building_area": {"path": ["source_info", "building_area"], "min": 0.0, "max": 2000.0, "delta": 100.0},
    "building_ht": {"path": ["source_info", "building_ht"], "min": 0.0, "max": 100.0, "delta": 5.0},
    "inland_or_coastal": {"path": ["source_info", "inland_or_coastal"], "values": ["inland", "coastal"]},
    "ws_calm_threshold": {"path": ["met_info", "ws_calm_threshold"], "min": 0.0, "max": 1.0, "delta": 0.1},
    # Receptor distance and terrain use special handlers
    "receptor_distance_min": {"kind": "receptor_distance_min", "min": 500.0, "max": 1500.0, "delta": 100.0},
    "receptor_distance_max": {"kind": "receptor_distance_max", "min": 2000.0, "max": 6000.0, "delta": 250.0},
    "receptor_terrain_min": {"kind": "receptor_terrain_min", "min": 0.0, "max": 50.0, "delta": 5.0},
    "receptor_terrain_max": {"kind": "receptor_terrain_max", "min": 0.0, "max": 50.0, "delta": 5.0},
}

# ============================================================
# CORE UTILITIES
# ============================================================

def make_sector_blocks(min_m: float, max_m: float, sectors: int = 16) -> List[List[float]]:
    return [[float(min_m), float(max_m)] for _ in range(sectors)]

def load_base_input(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Base input.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def run_atd(atd_exe: Path, input_json: Path, output_dir: Path, timeout_sec: int = 900) -> None:
    if not atd_exe.exists():
        raise FileNotFoundError(f"ATD.exe not found: {atd_exe}")
    if not input_json.exists():
        raise FileNotFoundError(f"Input not found: {input_json}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(atd_exe), str(input_json), str(output_dir)]
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False,
        timeout=timeout_sec
    )

    (output_dir / "ATD_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (output_dir / "ATD_stderr.log").write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        err_path = output_dir / "ATD_ENGINE.ERR"
        hint = f"\nSee: {err_path}" if err_path.exists() else ""
        raise RuntimeError(f"ATD returned code {proc.returncode}.{hint}\nSTDERR:\n{proc.stderr}")

    print("ATD finished OK. Outputs in:", output_dir)


def parse_xq_output(output_json: Path) -> List[Dict[str, Any]]:
    if not output_json.exists():
        raise FileNotFoundError(f"Missing output JSON: {output_json}")

    data = json.loads(output_json.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []

    for section in ("max_sector", "overall_site"):
        if section not in data:
            continue
        for item in data[section]:
            rows.append({
                "section": section,
                "title": item.get("title", ""),
                "statistic%": item.get("statistic%", ""),
                "EAB": item.get("EAB", ""),
                "LPZ": item.get("LPZ", "")
            })

    return rows


def set_by_path(payload: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = payload
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value


def get_receptor_bounds(payload: Dict[str, Any]) -> Dict[str, float]:
    dist = payload["receptor_info"]["distance"][0]
    terr = payload["receptor_info"]["terrain"][0]
    return {
        "distance_min": float(dist[0]),
        "distance_max": float(dist[1]),
        "terrain_min": float(terr[0]),
        "terrain_max": float(terr[1]),
    }


def set_receptor_distance(payload: Dict[str, Any], min_m: float, max_m: float) -> None:
    sectors = len(payload["receptor_info"]["distance"])
    payload["receptor_info"]["distance"] = make_sector_blocks(min_m, max_m, sectors)


def set_receptor_terrain(payload: Dict[str, Any], min_m: float, max_m: float) -> None:
    sectors = len(payload["receptor_info"]["terrain"])
    payload["receptor_info"]["terrain"] = make_sector_blocks(min_m, max_m, sectors)


def generate_values(cfg: Dict[str, Any]) -> List[Any]:
    if "values" in cfg:
        return list(cfg["values"])
    vmin = float(cfg["min"])
    vmax = float(cfg["max"])
    delta = float(cfg["delta"])
    values: List[float] = []
    v = vmin
    while v <= vmax + 1e-12:
        values.append(round(v, 12))
        v += delta
    return values


def apply_parameter(payload: Dict[str, Any], name: str, value: Any, base_bounds: Dict[str, float]) -> None:
    cfg = SENSITIVITY_CONFIG[name]
    if "path" in cfg:
        set_by_path(payload, cfg["path"], value)
        return

    kind = cfg.get("kind")
    if kind == "receptor_distance_min":
        set_receptor_distance(payload, float(value), base_bounds["distance_max"])
    elif kind == "receptor_distance_max":
        set_receptor_distance(payload, base_bounds["distance_min"], float(value))
    elif kind == "receptor_terrain_min":
        set_receptor_terrain(payload, float(value), base_bounds["terrain_max"])
    elif kind == "receptor_terrain_max":
        set_receptor_terrain(payload, base_bounds["terrain_min"], float(value))
    else:
        raise ValueError(f"Unknown parameter kind for {name}")


def resolve_met_files() -> List[Path]:
    if MET_FILES:
        return [Path(m) for m in MET_FILES]
    return [Path(p) for p in glob.glob(MET_GLOB)]


def safe_tag(value: Any) -> str:
    tag = str(value)
    return tag.replace(" ", "_").replace(".", "p").replace("/", "_")


def run_case(atd_path: Path, payload: Dict[str, Any], case_dir: Path, timeout_sec: int = 900) -> None:
    inputs_dir = case_dir / "inputs"
    outputs_dir = case_dir / "outputs"
    input_path = inputs_dir / "input.json"
    write_json(input_path, payload)
    run_atd(atd_path, input_path, outputs_dir, timeout_sec=timeout_sec)


def run_sensitivity(met_files: Iterable[Path], start_time: float) -> Path:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = CASE_ROOT_PATH / f"sierra_sensitivity_{run_tag}"
    summary_csv = run_dir / "xq_summary.csv"

    all_rows: List[Dict[str, Any]] = []
    base_input = load_base_input(BASE_INPUT_JSON)
    base_bounds = get_receptor_bounds(base_input)

    for met_file in met_files:
        met_file = met_file.resolve()
        if not met_file.exists():
            print(f"Skipping missing MET file: {met_file}")
            continue

        case_name = met_file.stem
        met_dir = run_dir / case_name

        if RUN_CONTROL:
            control_payload = copy.deepcopy(base_input)
            control_payload["control_info"]["scenario"] = "SIERRA_Sensitivity"
            control_payload["met_info"]["met_file"] = str(met_file)
            control_case_dir = met_dir / "control"
            run_case(ATD_PATH, control_payload, control_case_dir)

            output_json = control_case_dir / "outputs" / "ATD_ENGINE_OUT.JSON"
            rows = parse_xq_output(output_json)
            for r in rows:
                r["met_file"] = str(met_file)
                r["case_dir"] = str(control_case_dir)
                r["case_type"] = "control"
                r["parameter"] = ""
                r["value"] = ""
                all_rows.append(r)

        for param_name, cfg in SENSITIVITY_CONFIG.items():
            values = generate_values(cfg)
            for value in values:
                payload = copy.deepcopy(base_input)
                payload["control_info"]["scenario"] = "SIERRA_Sensitivity"
                payload["met_info"]["met_file"] = str(met_file)

                apply_parameter(payload, param_name, value, base_bounds)

                case_dir = met_dir / f"{param_name}_{safe_tag(value)}"
                run_case(ATD_PATH, payload, case_dir)

                output_json = case_dir / "outputs" / "ATD_ENGINE_OUT.JSON"
                rows = parse_xq_output(output_json)

                for r in rows:
                    r["met_file"] = str(met_file)
                    r["case_dir"] = str(case_dir)
                    r["case_type"] = "sensitivity"
                    r["parameter"] = param_name
                    r["value"] = value
                    all_rows.append(r)

    if all_rows:
        run_dir.mkdir(parents=True, exist_ok=True)
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "met_file",
                    "case_dir",
                    "case_type",
                    "parameter",
                    "value",
                    "section",
                    "title",
                    "statistic%",
                    "EAB",
                    "LPZ",
                ]
            )
            writer.writeheader()
            writer.writerows(all_rows)

            _, ranking_csv = analyze_sensitivity(all_rows, run_dir)
            write_report(
                run_dir=run_dir,
                met_files=list(met_files),
                start_time=start_time,
                end_time=time.perf_counter(),
                all_rows=all_rows,
                ranking_csv=ranking_csv,
            )

    return run_dir


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def analyze_sensitivity(all_rows: List[Dict[str, Any]], run_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    control_lookup: Dict[tuple, Dict[str, Any]] = {}
    for row in all_rows:
        if row.get("case_type") != "control":
            continue
        key = (
            row.get("met_file"),
            row.get("section"),
            row.get("title"),
            row.get("statistic%"),
        )
        control_lookup[key] = row

    detailed_rows: List[Dict[str, Any]] = []
    ranking_accum: Dict[tuple, Dict[str, float]] = {}

    for row in all_rows:
        if row.get("case_type") != "sensitivity":
            continue

        key = (
            row.get("met_file"),
            row.get("section"),
            row.get("title"),
            row.get("statistic%"),
        )
        control = control_lookup.get(key)
        if not control:
            continue

        eab = _safe_float(row.get("EAB"))
        lpz = _safe_float(row.get("LPZ"))
        eab0 = _safe_float(control.get("EAB"))
        lpz0 = _safe_float(control.get("LPZ"))

        delta_eab = eab - eab0 if eab is not None and eab0 is not None else None
        delta_lpz = lpz - lpz0 if lpz is not None and lpz0 is not None else None
        pct_eab = (delta_eab / eab0 * 100.0) if eab0 not in (None, 0.0) and delta_eab is not None else None
        pct_lpz = (delta_lpz / lpz0 * 100.0) if lpz0 not in (None, 0.0) and delta_lpz is not None else None

        detailed_rows.append({
            "met_file": row.get("met_file"),
            "parameter": row.get("parameter"),
            "value": row.get("value"),
            "section": row.get("section"),
            "title": row.get("title"),
            "statistic%": row.get("statistic%"),
            "control_EAB": eab0,
            "control_LPZ": lpz0,
            "EAB": eab,
            "LPZ": lpz,
            "delta_EAB": delta_eab,
            "delta_LPZ": delta_lpz,
            "pct_EAB": pct_eab,
            "pct_LPZ": pct_lpz,
        })

        param_key = (row.get("met_file"), row.get("parameter"))
        acc = ranking_accum.setdefault(param_key, {"sum_abs_pct": 0.0, "count": 0.0})
        if pct_eab is not None:
            acc["sum_abs_pct"] += abs(pct_eab)
            acc["count"] += 1.0
        if pct_lpz is not None:
            acc["sum_abs_pct"] += abs(pct_lpz)
            acc["count"] += 1.0

    detailed_csv: Optional[Path] = None
    ranking_csv: Optional[Path] = None

    if detailed_rows:
        detailed_csv = run_dir / "xq_sensitivity_results.csv"
        with detailed_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "met_file",
                    "parameter",
                    "value",
                    "section",
                    "title",
                    "statistic%",
                    "control_EAB",
                    "control_LPZ",
                    "EAB",
                    "LPZ",
                    "delta_EAB",
                    "delta_LPZ",
                    "pct_EAB",
                    "pct_LPZ",
                ]
            )
            writer.writeheader()
            writer.writerows(detailed_rows)

    if ranking_accum:
        ranking_rows: List[Dict[str, Any]] = []
        for (met_file, parameter), stats in ranking_accum.items():
            count = stats["count"] if stats["count"] else 0.0
            mean_abs_pct = (stats["sum_abs_pct"] / count) if count else None
            ranking_rows.append({
                "met_file": met_file,
                "parameter": parameter,
                "mean_abs_pct_change": mean_abs_pct,
                "n_values": int(count),
            })

        ranking_rows.sort(key=lambda r: (r["met_file"], -(r["mean_abs_pct_change"] or 0.0)))

        ranking_csv = run_dir / "xq_parameter_ranking.csv"
        with ranking_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["met_file", "parameter", "mean_abs_pct_change", "n_values"]
            )
            writer.writeheader()
            writer.writerows(ranking_rows)

    return detailed_csv, ranking_csv


def write_report(
    run_dir: Path,
    met_files: List[Path],
    start_time: float,
    end_time: float,
    all_rows: List[Dict[str, Any]],
    ranking_csv: Optional[Path],
) -> None:
    duration_sec = end_time - start_time
    total_cases = len({r.get("case_dir") for r in all_rows})
    control_cases = len({r.get("case_dir") for r in all_rows if r.get("case_type") == "control"})
    sensitivity_cases = len({r.get("case_dir") for r in all_rows if r.get("case_type") == "sensitivity"})

    top_params: List[str] = []
    if ranking_csv and ranking_csv.exists():
        rows = list(csv.DictReader(ranking_csv.open("r", encoding="utf-8")))
        rows.sort(key=lambda r: float(r.get("mean_abs_pct_change") or 0.0), reverse=True)
        top_params = [f"{r.get('parameter')} (mean_abs_pct_change={r.get('mean_abs_pct_change')})" for r in rows[:5]]

    report_path = run_dir / "sensitivity_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("SIERRA Sensitivity Report\n")
        f.write("===========================\n\n")
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"MET files: {len(met_files)}\n")
        for m in met_files:
            f.write(f"  - {m}\n")
        f.write("\n")
        f.write(f"Total cases run: {total_cases}\n")
        f.write(f"Control cases: {control_cases}\n")
        f.write(f"Sensitivity cases: {sensitivity_cases}\n")
        f.write(f"Duration (seconds): {duration_sec:.2f}\n")
        f.write(f"Duration (minutes): {duration_sec/60.0:.2f}\n\n")

        f.write("High-level takeaways (top mean absolute % change):\n")
        if top_params:
            for item in top_params:
                f.write(f"  - {item}\n")
        else:
            f.write("  - No ranking data available.\n")
        f.write("\n")
        f.write("Notes:\n")
        f.write("- Rankings are based on mean absolute percent change vs control across all X/Q outputs.\n")
        f.write("- Parameters with zero change may indicate insensitivity for the tested ranges,\n")
        f.write("  numerical rounding, or model rules that ignore the parameter under the chosen settings.\n")


if __name__ == "__main__":
    met_files = resolve_met_files()
    if not met_files:
        raise SystemExit("No MET files found. Update MET_FILES or MET_GLOB.")

    start_time = time.perf_counter()
    out_dir = run_sensitivity(met_files, start_time)
    print("Sensitivity run complete. Results in:", out_dir)
